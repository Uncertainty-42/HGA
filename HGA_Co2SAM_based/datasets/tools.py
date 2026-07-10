# datasets/tools.py
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide
from datasets.augmentation import weak_transforms, strong_transforms

class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, is_label=False, fill_value=0, return_valid_mask=False):
        """
        Perform proportional scaling and center-aligned padding (Resize and Square Padding)
        on the input data.

        This method is the core spatial alignment operator, responsible for uniformly mapping
        raw-resolution images or prior maps into the 1024x1024 SAM canonical space. It
        implements a "content-aware" processing strategy: for RGB images, normalization and
        channel permutation are performed; for priors (e.g., NAMLab/Depth), the original
        numerical range and 2D topology are strictly preserved.

        Computational Strategy:
        -----------------------
        1. Spatial mapping: compute a scale factor based on the original aspect ratio and
           resize the long side to 1024 pixels.
        2. Interpolation selection: if `is_label` is True, nearest-neighbor interpolation
           is enforced to prevent index or gradient artifacts.
        3. Zero-level padding: symmetric padding is applied on both sides of the short edge
           to produce a 1024x1024 square output.
        4. Geometric footprint: a valid mask (Valid Mask) is optionally generated to mask
           out padding-induced black borders in loss computation.

        Args:
            image (np.ndarray or torch.Tensor): Raw data to process.
                - Case A (RGB image): shape [H, W, 3], converted to [3, 1024, 1024] and
                  normalized to [0, 1].
                - Case B (Priors): shape [H, W], converted to [1024, 1024] while
                  preserving the original value distribution.
            is_label (bool): Whether the data is label/index data. If True, uses
                nearest-neighbor interpolation and skips 0-1 normalization.
            fill_value (float): Fill value for padded regions. For depth maps this can be
                set to an extreme negative value such as -100.
            return_valid_mask (bool): Whether to return the geometric validity mask.

        Returns:
            tuple (image_tensor, valid_mask_tensor):
                - image_tensor (torch.Tensor): Transformed tensor.
                    For RGB: [3, 1024, 1024]; for Priors: [1024, 1024].
                - valid_mask_tensor (torch.Tensor or None): Geometric validity mask.
                    Shape [1024, 1024], where 1 denotes original image pixels and 0
                    denotes padded regions.
        """
        # --- Defensive type conversion ---
        if hasattr(image, "detach"): # If Tensor, convert to Numpy
            image = image.detach().cpu().numpy()
        # ----------------------------

        # 1. Get original dimensions (compatible with both 2D and 3D inputs)
        og_h, og_w = image.shape[:2]
        
        # 2. Get the target size after resize expected by SAM
        target_h, target_w = self.transform.get_preprocess_shape(og_h, og_w, self.target_size)

        # 3. Perform interpolation resize (strictly distinguish interpolation modes)
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
        image_resized = cv2.resize(image, (target_w, target_h), interpolation=interp)

        # 4. Convert to Tensor (branch logic: distinguish between priors and RGB images)
        if len(image_resized.shape) == 2 or is_label:
            # [Prior channel]: directly convert to Tensor, preserving original values,
            # no dimension expansion [H, W]
            image_tensor = torch.from_numpy(image_resized)
        else:
            # [RGB image channel]: perform standard [3, H, W] conversion and 0-1 normalization
            image_tensor = self.to_tensor(image_resized)

        # 5. Compute padding parameters (strictly replicate original variable names and formulas)
        h, w = image_tensor.shape[-2:]
        max_dim = max(h, w) 
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        
        # 6. Apply padding
        image_padded = transforms.Pad(padding, fill=fill_value)(image_tensor)

        # 7. Synchronously generate validity mask (Valid Mask)
        if return_valid_mask:
            # Use an all-ones float32 matrix as the original mask
            valid_raw = np.ones((og_h, og_w), dtype=np.float32)
            # Resize (mask must use nearest-neighbor to preserve 0/1 boundaries)
            valid_resized = cv2.resize(valid_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            valid_tensor = torch.from_numpy(valid_resized) # [H W]
            # Padding (pad value fixed to 0, representing invalid regions)
            valid_mask_padded = transforms.Pad(padding, fill=0)(valid_tensor)
            
            return image_padded, valid_mask_padded

        return image_padded, None


def soft_transform(
        image: np.ndarray,
        priors: dict
    ):
    # 1. Assemble augmentation parameters
    # Note: tensors in priors must be converted back to numpy for albumentations processing
    aug_input = {"image": image}
    for k in ['namlab', 'depth']:
        data = priors.get(k)
        if data is not None:
            # Defensive conversion: convert to numpy regardless of whether it is a Tensor or ndarray
            aug_input[k] = data.detach().cpu().numpy() if hasattr(data, "detach") else data

    # 2. Apply state-synchronized weak augmentation
    weak_transformed = weak_transforms(**aug_input) # type: ignore
    image_weak = weak_transformed["image"]

    # 3. Update priors dictionary (backfill augmented results)
    for k in ['namlab', 'depth', 'valid_mask']:
        if k in weak_transformed:
            # Convert back to Tensor for downstream model usage
            priors[k] = torch.from_numpy(weak_transformed[k])

    # 4. Apply strong augmentation (strong augmentation typically only involves
    # color/noise and no geometric transformations, so priors do not need to be synchronized)
    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, image_strong, priors


def collate_fn(batch):
    """
    Data collation function for the validation set, supporting prior payloads (5 elements).
    """
    # 1. Unpack all 5 values
    images, gt_masks, image_paths, class_labels, priors_payloads = zip(*batch)
    
    # 2. Stack basic data
    images = torch.stack(images, 0)
    gt_masks = torch.from_numpy(np.stack(gt_masks, axis=0))
    class_labels = torch.stack(class_labels, 0)

    # 3. Process prior payload dictionary (priors_list: Tuple of Dicts)
    batched_priors = {}
    if priors_payloads[0] is not None and len(priors_payloads[0]) > 0:
        for key in priors_payloads[0].keys():
            values = [p[key] for p in priors_payloads]
            # If Tensor (e.g., namlab, depth), perform stack
            if isinstance(values[0], torch.Tensor):
                batched_priors[key] = torch.stack(values, dim=0)
            else:
                batched_priors[key] = values
    else:
        batched_priors = None
    
    # 4. Return the packed 5 values (paths are already tuples, no processing needed)
    return images, gt_masks, image_paths, class_labels, batched_priors


def collate_fn_soft(batch):
    """
    Batch collation function for the Co2SAM dual-teacher single-student architecture.
    Supports unpacking of 7 elements: [strong aug., weak aug., original, GT, path,
    categories, prior payload dictionary]
    """
    # 1. Perform unpacking (extract 7 independent tuples from N sample tuples)
    images_soft, images, images_origin, gt_mask, image_path, categories, priors_payloads = zip(*batch)

    # 2. Stack basic image data (maintaining the original refactored logic)
    # images and images_soft are SAM-size tensors [3, 1024, 1024]
    # images_origin are original-size numpy arrays [H, W, 3]
    images = torch.stack(images)
    # images_origin = np.stack(images_origin)
    images_soft = torch.stack(images_soft)

    # 3. Process the 7th element: prior payload dictionary (priors_payloads)
    # priors_payloads is a tuple of N dictionaries: ({k:v}, {k:v}, ...)
    # Goal: transform it into {k: Batched_Tensor}
    batched_priors = {}
    
    # Use the keys from the first sample's dictionary as a reference
    if priors_payloads[0] is not None:
        for key in priors_payloads[0].keys():
            # Extract all data corresponding to this key across the entire batch
            values = [p[key] for p in priors_payloads]

            # Strict check: no prior field should be missing
            for i, val in enumerate(values):
                if val is None:
                    raise RuntimeError(
                        f"[Collate Error] Image {image_path[i]} missing prior field '{key}'. "
                        f"Please check the corresponding PriorsManager data integrity."
                    )
            
            # All values present and are Tensors; stack normally
            if all(isinstance(v, torch.Tensor) for v in values):
                batched_priors[key] = torch.stack(values, dim=0)
            else:
                raise TypeError(f"[Collate Error] Non-Tensor value found in prior field '{key}'.")
            
            # # Automatic stacking logic:
            # # If Tensor (e.g., valid_mask, namlab), execute stack to add batch dimension
            # if isinstance(values[0], torch.Tensor):
            #     batched_priors[key] = torch.stack(values, dim=0)
            # else:
            #     # If None or other non-Tensor data, keep as list/tuple
            #     batched_priors[key] = values
    else:
        # If the entire batch has no priors, return None or empty dict
        batched_priors = None

    return images_soft, images, images_origin, gt_mask, image_path, categories, batched_priors


def collate_fn_(batch):
    return zip(*batch)

