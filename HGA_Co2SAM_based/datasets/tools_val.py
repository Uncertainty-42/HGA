# datasets/tools_val.py
import cv2
import torch
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide

class Resize:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, bboxes=None, visual=False, is_label=False):
        """
        Perform proportional scaling on images or priors (without padding).

        Args:
            image (np.ndarray): Input data [H, W, 3] or [H, W].
            is_label (bool): When True, use nearest-neighbor interpolation without
                0-1 normalization, preserving original indices/values.
        """
        # Get original dimensions (compatible with 2D and 3D inputs)
        if is_label:
            og_h, og_w = image.shape
            # Processing logic for priors/labels
            # 1. Compute scaled target dimensions (based on SAM's ResizeLongestSide logic)
            target_h, target_w = self.transform.get_preprocess_shape(og_h, og_w, self.target_size)
            # 2. Force nearest-neighbor interpolation
            image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            # 3. Directly convert to Tensor without 0-1 normalization or channel permutation
            return torch.from_numpy(image_resized)
        else:
            # --- Preserve the original RGB image processing logic ---
            image = self.transform.apply_image(image)
            image = self.to_tensor(image)
            return image

