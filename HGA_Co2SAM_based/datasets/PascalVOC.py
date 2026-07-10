# datasets/PascalVOC.py
import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft
from datasets.tools_val import Resize
from PIL import Image

from modules.tools.priors_manager import PriorsManager # type: ignore

class PascalVOCDataset(Dataset):
    def __init__(self, cfg, root_dir, transform=None, training=False, if_self_training=False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.dataset_cfg = self.cfg.datasets.PascalVOC
        self.transform = transform
        self.if_self_training = if_self_training
        self.training = training

        self.is_testing = cfg.get("is_testing", False)

        # --- Core Definition ---
        # 0: background, 1: aeroplane, ... 20: tvmonitor (matching the mapping logic in eval_utils)
        self.PASCAL_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # 1. Select the list file
        if training:
            list_file = os.path.join(
                root_dir, 
                self.dataset_cfg.get("train_aug_txt", "ImageSets/SegmentationAug/standard_train_aug.txt")
            )
        elif not self.is_testing:
            list_file = os.path.join(
                root_dir, 
                self.dataset_cfg.get("val_txt", "ImageSets/SegmentationAug/standard_val.txt")
            )
        else:
            list_file = os.path.join(
                root_dir, 
                self.dataset_cfg.get("test_txt", "ImageSets/SegmentationAug/test.txt")
            )

        # 2. Read IDs
        with open(list_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # 3. Define paths
        self.image_dir = os.path.join(self.root_dir, "JPEGImages" if not self.is_testing else "VOC2012test/JPEGImages")
        # Note: Mask is usually in SegmentationClass or SegmentationClassAug
        self.mask_dir = os.path.join(
            self.root_dir, 
            self.dataset_cfg.get("segment_root", "SegmentationClassAug")
        ) 
        # [Critical] Ensure the Annotations path is correct
        self.annotation_dir = os.path.join(self.root_dir, self.dataset_cfg.get("annotation_root", "Annotations"))

        # Verify path existence (fact-check)
        if not os.path.exists(self.annotation_dir):
            print(f"WARNING: Annotation dir not found at {self.annotation_dir}")

        # Whether training or validation, if priors are enabled, inject the manager with cfg and the current transform
        self.priors_enabled = self.cfg.get("priors", {}).get("enabled", False)
        if self.priors_enabled:
            self.priors_manager = PriorsManager(self.cfg, transform=None)
        if training:
            random.shuffle(self.image_ids)
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Get ID
        image_id = self.image_ids[idx]

        # 2. Construct path
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, image_id + '.png')
        
        # 3. Read image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[WARN] Error loading image: {image_path}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        image_origin = image.copy()

        # 4. Use PIL to read mask index values instead of OpenCV reading grayscale values
        try:
            if os.path.exists(mask_path):
                # PIL opening a P-mode image directly yields indices
                mask_pil = Image.open(mask_path)
                gt_mask = np.array(mask_pil).astype(np.uint8)
            else:
                if not self.is_testing:
                    print(f"[WARN] Mask not found at {mask_path}, using blank mask.")
                gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        except Exception as e:
            print(f"[WARN] Error loading mask: {mask_path}: {e}")
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 5. Extract image-level labels from GT mask
        class_label = torch.zeros(21, dtype=torch.float)
        
        unique_classes = np.unique(gt_mask)
        
        for cls_idx in unique_classes:
            # Ignore background (0) and ignored region (255)
            # Note: Indices read by PIL should never contain intermediate values like 147 unless the mask file is corrupted
            if cls_idx > 0 and cls_idx < 255:
                if cls_idx <= 20:
                    class_label[int(cls_idx)] = 1.0
                else:
                    # If there are still strange values after fixing, print a warning
                    if idx < 10: 
                        print(f"[WARN Label] Unexpected class index {cls_idx} in {image_id}")
        
        # In training mode, use soft_transform to generate real weak/strong augmentation pairs
        if self.training:
            if self.priors_enabled:
                # Get prior payload (7th element)
                # image_origin is the original image before 1024 resizing, used for feature computation
                priors = self.priors_manager.get_payload(image_id, image_origin)
            else:
                priors = {}
            
            # Call soft_transform from tools.py (internally calls augmentation.py)
            # Both image_weak_np and image_strong_np are numpy arrays
            image_weak_np, image_strong_np, priors = soft_transform(image, priors)
            
            # Resize and convert to tensor for weak and strong images respectively
            if self.transform:
                image_weak, priors['valid_mask'] = self.transform(image_weak_np, return_valid_mask=True)
                image_strong, _ = self.transform(image_strong_np)
                if priors.get('namlab') is not None:
                    priors['namlab'], _ = self.transform(priors['namlab'], is_label=True)
                if priors.get('depth') is not None:
                    priors['depth'], _ = self.transform(priors['depth'], is_label=False)
            else:
                # Fallback logic (theoretically won't trigger since load_datasets passes transform)
                import torchvision.transforms.functional as F_vis
                image_weak = F_vis.to_tensor(image_weak_np)
                image_strong = F_vis.to_tensor(image_strong_np)

            
            
            # Return the real image_strong, no longer a copy
            return image_weak, image_strong, image_origin, gt_mask, image_path, class_label, priors

        else:
            # Validation/test mode: only apply basic transforms
            priors = {}
            if self.priors_enabled:
                # 1. Extract raw prior data (NAMLab/Depth, etc.)
                priors = self.priors_manager.get_payload(image_id, image_origin)
            if self.transform:
                # 2. Resize RGB image
                transformed_image = self.transform(image)
                
                # 3. Resize prior maps (using Resize's new is_label mode to ensure nearest-neighbor
                #    interpolation and values not being normalized)
                if priors.get('namlab') is not None:
                    priors['namlab'] = self.transform(priors['namlab'], is_label=True)
                if priors.get('depth') is not None:
                    # Note: Although Depth is not an indexed label, in the SAM framework it is usually
                    # handled in its original range; here we follow the label mode for resizing
                    priors['depth'] = self.transform(priors['depth'], is_label=True)
                # Validation mode uses Resize instead of ResizeAndPad; the return value is single, not a tuple
            else:
                import torchvision.transforms.functional as F_vis
                transformed_image = F_vis.to_tensor(image)

            return transformed_image, gt_mask, image_path, class_label, priors
        

def load_datasets_soft(cfg, img_size, fabric):
    rand_seed = cfg.rand_seed
    g = torch.Generator()
    g.manual_seed(rand_seed + fabric.global_rank)
   
    transform = ResizeAndPad(img_size)
    transform_val = Resize(img_size)
    val = PascalVOCDataset(
        cfg,
        root_dir=cfg.datasets.PascalVOC.root_dir,
        transform=transform_val,
    )
    soft_train = PascalVOCDataset(
        cfg,
        root_dir=cfg.datasets.PascalVOC.root_dir,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        generator=g
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
        generator=g
    )
    return soft_train_dataloader, val_dataloader