"""
tools/priors_manager.py
Hierarchical & Geometric Priors I/O Manager
=======================================

This module handles I/O management, cache routing, and automated production
scheduling for all experimental prior data.

Functional Scope:
--------------------------
Acting as a "transfer station" between the Dataset and the engine, the Manager
implements an atomic "Check-and-Gen" caching logic: read if available, generate
if missing. It abstracts away the complex Stage A-B-C scheduling of NAMLab,
providing a one-click data retrieval service via image_id.

Algorithm Principle:
----------------------------
1. **Dynamic Pathing**: Dynamically constructs subdirectories based on
   target_level, strategy, and threshold in the configuration, achieving
   experiment parameter isolation.
2. **Atomic Persistence**: Stores uint16 index maps in .npy format to
   ensure boundary precision.
3. **Payload Packaging**: Packages discrete prior data (NAMLab, Depth, etc.)
   into dictionary containers, supporting adaptive reading by downstream
   modules.

Call Flow:
-------------------
Dataset.__getitem__ -> PriorsManager.get_payload
    -> _check_and_gen_namlab (If missing -> namlab.core.run_namlab_refinement)
    -> _apply_alignment_transformation (Placeholder)
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import the engine core
from modules.prior_refinement.namlab.core import run_namlab_refinement


class PriorsManager:
    """
    Prior data management class. Responsible for managing the lifecycle and
    I/O of NAMLab index maps, depth maps, and other prior data.
    """
    def __init__(self, config: Any, transform: Any = None,
                 dataset: str = "voc", is_training: Optional[bool] = None):
        """
        Initialize the manager.

        Args:
            config (Any): Global configuration object (Box).
            transform: Transformer for resizing and padding images.
                       If None, no transformation is applied.
            is_training (Optional[bool]): For COCO dataset, whether in
                                          training mode.
        """
        self.cfg = config
        self.nam_cfg = config.priors.namlab
        # Receive the ResizeAndPad instance from Dataset
        self.transform = transform
        self.dataset = dataset
        if self.dataset == "coco":
            assert is_training is not None
            self.is_training = is_training

        if self.nam_cfg.get('enabled', False):
            if self.nam_cfg.get("purification_enabled", True) is False:
                print(
                    "[Config] ⚠️ PriorManager: NAMLab region purification "
                    "disabled! Using raw NAMLab target level"
                )
            else:
                print(
                    "[Config] 🥼 PriorManager: NAMLab region purification "
                    "enabled! Will further eliminate small regions based on "
                    "the raw NAMLab target level"
                )

    def _get_namlab_sub_dir(self) -> Path:
        """
        Dynamically derive and ensure the existence of the NAMLab cache
        subdirectory based on current experimental parameters.

        Algorithm Principle:
        ---------
        Parses the config.priors.namlab.paths.cache_template template, mapping
        parameters such as level, strategy, and threshold to concrete paths,
        achieving cache isolation across different merge depths.

        Returns:
            Path: Path object for the cache subdirectory.
        """
        if self.nam_cfg.get("purification_enabled", True):
            dir_name = self.nam_cfg.paths.cache_template.format(
                level=self.nam_cfg.target_level,
                strategy=self.nam_cfg.area_constraint.strategy,
                threshold=self.nam_cfg.area_constraint.threshold
            )
        else:
            dir_name = self.nam_cfg.paths.cache_template.format(
                level=self.nam_cfg.target_level
            )

        if self.dataset == "coco":
            sub_dir = "train" if self.is_training else "val"
            target_dir = (Path(self.nam_cfg.paths.cache_root)
                          / dir_name / sub_dir)
        else:
            target_dir = Path(self.nam_cfg.paths.cache_root) / dir_name

        if not target_dir.exists():
            # Non-silent creation: only triggered once when directory is
            # missing
            target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def get_payload(self, image_id: str,
                    image_origin: np.ndarray) -> Dict[str, Any]:
        """
        Assemble and return an "adaptive prior payload" dictionary.

        Functional Positioning:
        ---------
        This interface is the sole entry point for the Dataset. It determines
        which priors are activated based on config-driven switches, and
        packages the processed Tensors into a dictionary to ensure strong
        extensibility of the data flow.

        Args:
            image_id (str): Image identifier.
            image_origin (np.ndarray): Original image.

        Returns:
            Dict: Prior payload dictionary, e.g.,
                  {'namlab': Tensor, 'depth': None}.
        """
        payload = {}

        # Basic geometric payload: whether priors are enabled or not, always
        # generate the valid region mask. This ensures that even under the
        # baseline, the loss function can mask out padding regions.
        _, valid_mask = self._apply_alignment_transformation(
            data=image_origin, is_label=False, return_valid_mask=True
        )
        payload['valid_mask'] = valid_mask  # [1, 1024, 1024] Tensor

        # 1. NAMLab Prior (logic-driven activation)
        if self.nam_cfg.get('enabled', False):
            mask = self._check_and_gen_namlab(image_id, image_origin)
            # 1. Invoke and unpack the transformed data and valid mask
            mask_aligned, _ = self._apply_alignment_transformation(
                data=mask, is_label=True, fill_value=0,
                return_valid_mask=True
            )

            # 2. Store data (Note: mask_aligned is already a Tensor; just
            #    squeeze and convert to long)
            payload['namlab'] = (
                mask_aligned.squeeze(0).long()
                if (hasattr(mask_aligned, 'squeeze')
                    and not isinstance(mask_aligned, np.ndarray))
                else mask_aligned
            )
        else:
            payload['namlab'] = None

        # 2. Geometric Depth Prior
        # Strictly abide by the switch contract: only load raw data when
        # enabled is True
        if self.cfg.priors.get('depth', {}).get('enabled', False):
            try:
                # 1. Locate and load the raw depth map
                if self.dataset == "coco":
                    sub_dir = "train" if self.is_training else "val"
                    depth_path = (Path(self.nam_cfg.paths.raw_depth_dir)
                                  / sub_dir / f"{image_id}.npy")
                else:
                    depth_path = (Path(self.nam_cfg.paths.raw_depth_dir)
                                  / f"{image_id}.npy")

                if not depth_path.exists():
                    raise FileNotFoundError(
                        f"Depth file not found: {depth_path}"
                    )

                # Preserve raw values (absolutely do NOT normalize here)
                depth_raw = np.load(str(depth_path))

                # 2. Spatial alignment (is_label=False ensures smooth
                #    continuity of depth values)
                depth_aligned, _ = self._apply_alignment_transformation(
                    data=depth_raw, is_label=False, fill_value=0,
                    return_valid_mask=True
                )

                # 3. Store data
                payload['depth'] = (
                    depth_aligned.squeeze(0).float()
                    if (hasattr(depth_aligned, 'squeeze')
                        and not isinstance(depth_aligned, np.ndarray))
                    else depth_aligned
                )

            except Exception as e:
                print(
                    f"⚠️ [PriorsManager Error] Failed to process Depth "
                    f"for ID: {image_id}"
                )
                print(
                    f"   Exception: {type(e).__name__} | Info: {str(e)}"
                )
                payload['depth'] = None
        else:
            payload['depth'] = None

        return payload

    def _check_and_gen_namlab(self, image_id: str,
                              image_origin: np.ndarray) -> np.ndarray:
        """
        Execute cache checking and atomic production of NAMLab index maps.

        Functional Logic:
        ---------
        1. Locate: check whether {cache_dir}/{image_id}.npy exists.
        2. Hit: directly return the in-memory numpy array.
        3. Miss: read .pt from raw_pt_dir, read .npy from raw_depth_dir,
           and invoke the engine to generate.

        Args:
            image_id (str): Image ID.
            image_origin (np.ndarray): Original-size image (H, W, 3), used
                                       for mean computation.

        Returns:
            np.ndarray: Refined index map (uint16).
        """

        cache_path = self._get_namlab_sub_dir() / f"{image_id}.npy"

        if cache_path.exists():
            return np.load(str(cache_path))

        try:
            # Locate raw materials (PT data and NPY depth tensor)
            if self.dataset == "coco":
                sub_dir = "train" if self.is_training else "val"
                pt_path = (Path(self.nam_cfg.paths.raw_pt_dir)
                           / sub_dir / f"{image_id}.pt")
                depth_path = (Path(self.nam_cfg.paths.raw_depth_dir)
                              / sub_dir / f"{image_id}.npy")
            else:
                pt_path = (Path(self.nam_cfg.paths.raw_pt_dir)
                           / f"{image_id}.pt")
                depth_path = (Path(self.nam_cfg.paths.raw_depth_dir)
                              / f"{image_id}.npy")

            if not pt_path.exists() or not depth_path.exists():
                raise FileNotFoundError(
                    f"Missing materials for {image_id}"
                )

            pt_data = torch.load(str(pt_path), map_location='cpu')
            depth_raw = np.load(str(depth_path))

            refined_mask = run_namlab_refinement(
                pt_data=pt_data,
                img_rgb=image_origin,
                depth_raw=depth_raw,
                config=self.cfg
            )

            # Persist to storage to avoid recomputation in subsequent epochs.
            # Use temp file + atomic replace to avoid write conflicts when
            # num_workers > 0.
            tmp_path = cache_path.with_suffix(f".tmp_{os.getpid()}")
            try:
                np.save(str(tmp_path), refined_mask)
                os.replace(str(tmp_path) + ".npy", str(cache_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            return refined_mask

        except Exception as e:
            # Non-silent failure principle: print detailed state-machine
            # diagnostics
            print(
                f"⚠️ [PriorsManager Error] Refinement failed for "
                f"ID: {image_id}"
            )
            print(
                f"   Exception: {type(e).__name__} | Info: {str(e)}"
            )
            return np.zeros(image_origin.shape[:2], dtype=np.uint16)

    def _apply_alignment_transformation(
        self,
        data: np.ndarray,
        is_label: bool = True,
        fill_value: float = 0,
        return_valid_mask: bool = False
    ) -> tuple:
        """
        Spatially aligning prior data.

        Functional Positioning:
        --------
        Scaling and padding logic synchronized with
        datasets.tools.ResizeAndPad to ensure
        pixel-level alignment between prior masks and SAM's Image Embedding.

        Args:
            data (np.ndarray): Original-size data.
            target_size: Target size parameters.
            is_label (bool): Whether to use nearest-neighbor interpolation.

        Returns:
            tuple(np.ndarray, np.ndarray | None): [0] Transformed data;
                [1] Validity mask, or None.
        """
        if self.transform is None:
            return data, None

        # Directly invoke the fine-tuned ResizeAndPad instance
        return self.transform(
            data,
            is_label=is_label,
            fill_value=fill_value,
            return_valid_mask=return_valid_mask
        )