# _Optimization_Workspace/tools/visualize.py

import os
from typing import Optional
import torch
import numpy as np
from pathlib import Path
import json
import shutil
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from lightning import Fabric

# Matplotlib is imported inside drawing methods to avoid server-side GUI errors on import.

VIRIDIS_SPEC = [(0.0, "#440154"), (0.25, "#3b528b"), (0.5, "#21918c"), (0.75, "#5ec962"), (1.0, "#fde725")]  # Viridis: purple-green
GRAY_SPEC = [(0.0, "black"), (1.0, "white")]  # Pure black-and-white

def _generate_voc_palette(n=256):
    """
    Standard PASCAL VOC color map generation.
    Returns: np.ndarray of shape (256, 3) with RGB values.
    """
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        label = i
        r, g, b = 0, 0, 0
        for j in range(8):
            r |= ((label >> 0) & 1) << (7 - j)
            g |= ((label >> 1) & 1) << (7 - j)
            b |= ((label >> 2) & 1) << (7 - j)
            label >>= 3
        palette[i] = [r, g, b]
    return palette

# Use standard function to generate palette (full 0-255 coverage)
VOC_PALETTE = _generate_voc_palette(256)

# Standard VOC 21-class ordering, used to physically align palette indices
standard_voc_labels = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class Visualizer:
    """
    Global Singleton Visualization Service.

    This module provides a context-aware, configuration-driven visualization probe system.
    It is designed as a global singleton, accessible anywhere in the project via
    `from ... import VISUALIZER`.
    Beyond image visualization, it also supports recording metadata (Metrics Snapshot)
    associated with the current visualization frame.

    **Data Management:**
        - **Global Dictionary:** Provides `VISUALIZER.metrics` (dict) for direct read/write by users.
        - **Automatic Sync:** Provides `VISUALIZER.sync()` method, which performs type-safety checks
          and atomically writes `metrics.json`.
        - **Checkpoint-Resume:** Automatically loads existing `metrics.json` on initialization,
          supporting experiment recovery.

    **Core Workflow:**
        1. **Initialization (in main script):**
           `VISUALIZER.initialize(cfg, run_paths.viz)`

        2. **State Update (in train/eval loop):**
           `VISUALIZER.update_state(epoch, iter, mode='train')`

        3. **Probe Injection (in model/modules):**
           if VISUALIZER.is_active("probe_name"):
               # a. Caller is responsible for data preprocessing
               processed_data = VISUALIZER.preprocess_as_heatmap(tensor)
               # b. Caller commands the service to execute drawing
               VISUALIZER.draw_heatmap((processed_data,), "probe_name")

    **Design Principles:**
        - **State Encapsulation:** All I/O details (paths, filenames, counters) are internally managed.
        - **Configuration-Driven:** The `is_active` method determines whether to execute by checking
          the config `probes` list and frequency policy.
        - **Separation of Concerns:** The caller handles data transformation; this class handles
          drawing and saving.
    """
    def __init__(self):
        """
        Initialize the basic state of the Visualizer instance.

        Note:
            The instance is not ready for use after instantiation; `initialize()` must be
            called first to inject configuration and paths.
        """
        self.is_initialized = False
        self.cfg = None
        self.root_save_dir = None

        self.current_epoch = -1
        self.current_iter = -1
        self.mode = 'idle'

        self.step_counter = 0
        self.current_save_dir = None

        # Metrics Management
        self.metrics = {}
        self.metrics_file = None

        self.fabric = None

    def _print(self, *args):
        """
        Print, compatible with both fabric and non-fabric environments.
        """
        if self.fabric is not None:
            self.fabric.print(args)
        else:
            print(args)

    def initialize(self, cfg, root_save_dir: str, fabric: Optional[Fabric]):
        """
        Called at the start of the main training script to inject configuration
        and set up the root directory for the visualizer.

        Args:
            cfg (Box): Global configuration object. Must contain `cfg.visual` node,
                       which should have items such as `probes` (list), `policy` (dict), etc.
            root_save_dir (str): Root directory path for saving visualization results
                                 (usually provided by `run_paths.viz`).
                                 The system converts the path to a `pathlib.Path` object.
            fabric (L.Fabric, optional): Fabric handle, used for multi-GPU confusion matrix synchronization.
        """
        self.cfg = cfg.visual
        self.root_save_dir = Path(root_save_dir)
        self.is_initialized = True
        self.fabric = fabric
        self._print(f"[Visualizer] Initialized. Saving to: {self.root_save_dir}")
        # print(cfg)
        self._print(f"[Visualizer] Probes enabled: {self.cfg.probes}")

        # Configure Metrics file path
        self.metrics_file = self.root_save_dir / "metrics.json"

        # Attempt to load existing data (checkpoint-resume support)
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                self._print(f"[Visualizer] Loaded existing metrics from: {self.metrics_file}")
            except Exception as e:
                self._print(f"[Visualizer] Warning: Failed to load existing metrics: {e}")

    def update_state(self, epoch: int, iter: int, mode: str, silent: bool = False):
        """
        Called by the training or evaluation loop to update the current context state.

        This method performs no I/O operations; it only updates internal state registers so that
        subsequent calls to `is_active` and path generation functions can obtain the correct
        epoch and iteration information.

        Args:
            epoch (int): Current epoch index (usually starting from 1).
            iter (int): Iteration index within the current epoch.
            mode (str): Current run mode, typically 'train' or 'eval'.
                        This mode affects the save directory prefix (e.g., 'Eval_').
            silent (bool): Whether to update silently without printing status. Defaults to False (print).
        """
        self.current_epoch = epoch
        self.current_iter = iter
        self.mode = mode
        if not silent:
            self._print(f"epoch {epoch}, iter {iter}, mode {mode}")

    def is_active(self, name: str) -> bool:
        """
        Check whether the specified probe should be activated at this moment.

        This is the **Gatekeeper** for all visualization operations. It performs four levels of checks:
        1. **Fabric:** Whether fabric is provided; if so, whether the current process is the main process.
        2. **Global Switch:** Whether `cfg.visual.enabled` is True.
        3. **Whitelist Check:** Whether the requested `name` is in the `cfg.visual.probes` list.
        4. **Frequency Control:** Uses `_should_draw_now()` to determine whether the current iteration
           matches the sampling policy.

        Side Effects:
            If True is returned, this method also calls `_prepare_current_step()`,
            ensuring that the corresponding output directory is created and the step counter is reset.

        Args:
            name (str): Unique identifier name of the probe (e.g., 'Vis/Logits_Heatmap').
                        Special macro: '@all' — skips the whitelist check and only performs
                        the frequency check. i.e., does not check whether a specific name is in
                        cfg.probes; returns True as long as the frequency check passes, otherwise False.

        Returns:
            bool: True if visualization should be executed now; otherwise False.
        """
        if self.fabric and self.fabric.global_rank != 0:
            return False

        assert self.cfg, "[Error] In tools/visualize.py Visualizer.is_active: self.cfg cannot be None!"
        if not self.is_initialized or not self.cfg.enabled:
            return False

        # 1. Configuration check
        if name != '@all' and name not in self.cfg.probes:
            return False

        # 2. Frequency check
        if not self._should_draw_now():
            return False

        # 3. Preparation work (directory creation & counter reset)
        self._prepare_current_step()
        return True

    def _should_draw_now(self) -> bool:
        """Internal logic: determine whether it is time to draw based on the three-phase strategy."""
        assert self.cfg, "[Error] In tools/visualize.py Visualizer._should_draw_now: self.cfg cannot be None!"
        # print(f"[Debug] self.mode: {self.mode}, self.cfg.get('eval_enabled', False): {self.cfg.get('eval_enabled', False)}")
        if self.mode == 'eval':
            if not self.cfg.get('eval_enabled', False):
                return False

        policy = self.cfg.policy
        if self.current_epoch <= 1:
            # print(f"{self.current_iter <= policy.warmup_steps}, {self.current_iter % policy.warmup_freq == 0}, {self.current_iter % policy.epoch1_freq == 0}")
            if self.current_iter <= policy.warmup_steps:
                return self.current_iter % policy.warmup_freq == 0
            else:
                return self.current_iter % policy.epoch1_freq == 0
        else:
            return self.current_iter % policy.later_freq == 0

    def _prepare_current_step(self):
        """Internal function: generate and create the save directory for the current step, and reset the counter."""
        prefix = "Eval_" if self.mode == 'eval' else ""
        step_dir_name = f"{prefix}Epoch_{self.current_epoch}_Iter_{self.current_iter}"
        assert self.root_save_dir, "[Error] In tools/visualize.py Visualizer._prepare_current_step: self.root_save_dir cannot be None!"

        new_save_dir = self.root_save_dir / step_dir_name

        if new_save_dir != self.current_save_dir:
            self.current_save_dir = new_save_dir
            self.step_counter = 0
            os.makedirs(self.current_save_dir, exist_ok=True)

    def _get_save_path(self, name: str, extension: str) -> str:
        """
        [Private Core] Compute and return a unique artifact save path for the current step.

        This function is the I/O foundation for all drawing commands. It does not perform
        any actual write operations; it only generates a safe, non-colliding file path string
        based on the current context (epoch, iter) and internal counter.

        Args:
            name (str): The base name portion of the file.
            extension (str): File extension (without dot), e.g., 'png', 'json'.

        Returns:
            str: Complete absolute file path string.

        Raises:
            RuntimeError: Raised if `is_active()` was not successfully called before this method
                          for context preparation (fail-fast principle).
        """
        if not self.current_save_dir:
            raise RuntimeError(
                "[Visualizer] Save directory not prepared. "
                "A `draw_...` method was likely called without "
                "being guarded by `if VISUALIZER.is_active(...):`."
            )

        filename = f"{self.step_counter:03d}_{name}.{extension}"
        full_path = self.current_save_dir / filename
        self.step_counter += 1
        return str(full_path)


    def sync(self):
        """
        Safely synchronize the global metrics dictionary to disk.

        Execution steps:
        1. Content Check: If metrics is None or an empty dict, silently return without any action.
        2. Type Check: Ensure the dictionary contains only JSON-safe basic types;
           illegal types will trigger a ValueError.
        3. Atomic Write: Write to a temporary file first, then perform a rename operation
           to prevent data corruption from interruptions.
        """
        if not self.is_initialized:
            return

        # Content check
        if self.metrics is None or self.metrics == {}:
            return

        # 2. Type pre-check
        try:
            self._validate_json_serializable(self.metrics)
        except TypeError as e:
            raise TypeError(f"[Visualizer] Sync failed: {e}")

        # 3. Atomic write
        assert self.metrics_file is not None, "self.metrics_file cannot be None!"
        temp_file = self.metrics_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)

            # Atomic Replace
            # os.replace is atomic on POSIX systems, and also atomic on Windows (Python 3.3+)
            os.replace(temp_file, self.metrics_file)
        except Exception as e:
            if temp_file.exists():
                os.remove(temp_file)
            raise IOError(f"[Visualizer] Failed to write metrics file: {e}")

    def _validate_json_serializable(self, obj, path="root"):
        """
        Recursively check whether the object and its sub-elements contain only JSON-safe basic types.

        This method is used as a data integrity sentinel check before writing to disk. If any
        unsupported data type (e.g., custom objects, NumPy arrays, Tensors, etc.) is found,
        an exception is immediately raised to prevent generating a corrupted JSON file.

        Args:
            obj (Any): The object to inspect. Supported types include str, int, float, bool, None, list, dict.
            path (str, optional): The current traversal path string (for error tracking). Defaults to "root".

        Raises:
            TypeError: Raised when encountering a data type not supported by JSON.
        """
        if obj is None:
            return

        # Primitive type whitelist
        if isinstance(obj, (str, int, float, bool)):
            return

        # Container type recursive check
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                self._validate_json_serializable(item, path=f"{path}[{i}]")
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise TypeError(f"Key at '{path}' must be string, got {type(key)}")
                self._validate_json_serializable(value, path=f"{path}.{key}")
            return

        # If none of the above, raise an exception
        raise TypeError(f"Object at '{path}' has unsupported type: {type(obj)}. Allowed: str, int, float, bool, None, list, dict.")

    # =========================================================================
    # Data Pre-processing Toolbox
    # On-demand Implementation
    # =========================================================================
    # This area is for adding helper functions designed to convert tensors
    # from inside the model into standard formats suitable for visualization
    # (e.g., NumPy arrays).
    #
    # Convention:
    # - Function naming: `preprocess_as_<target_format>(self, tensor, ...)`
    # - Example: `preprocess_as_heatmap(self, feature_map)` -> np.ndarray
    # - Responsibility: Only data conversion (detach, cpu, numpy, normalize),
    #   no involvement in drawing.

    # =========================================================================
    # Drawing Commands
    # On-demand Implementation
    # =========================================================================
    # This area is for adding concrete drawing/saving methods. Each method should
    # follow the conventions below:
    #
    # Convention:
    # - Function naming: `draw_<artifact_type>(self, data_tuple, name, ...)`
    # - `data_tuple` (tuple): Tuple containing all data to be visualized.
    #                         Even if there is only one element, it should be
    #                         wrapped as `(element,)`.
    # - `name` (str): Core identifier used to generate the filename
    #                 (without prefix or suffix).
    # - Internal workflow: 1. Call self._get_save_path() to obtain the path.
    #                       2. Execute drawing and saving (e.g., plt.savefig, cv2.imwrite).


    def draw_semseg_error_analysis(self, data_tuple, name):
        """
        Draw a 2x2 grid for semantic segmentation error analysis.

        Args:
            data_tuple: (image_np, pred_mask, gt_mask)
                - image_np: [H, W, 3] RGB, uint8
                - pred_mask: [H, W] predicted mask (0-20), uint8
                - gt_mask: [H, W] ground truth mask (0-20, 255=Ignore), uint8
            name: Filename identifier (without suffix)

        Output Layout:
            [GT Green]    [Pred Cyan]
            [Overflow Red][Missing Blue]
        """
        import cv2
        import numpy as np

        image, pred, gt = data_tuple

        # Ensure consistent dimensions (required for OpenCV drawing)
        if image.shape[:2] != pred.shape:
            image = cv2.resize(image, (pred.shape[1], pred.shape[0]))

        # Helper function: draw semi-transparent overlay
        def apply_overlay(bg_img, mask, color_rgb):
            # mask: boolean map
            # color_rgb: tuple (R, G, B)
            if not np.any(mask):
                return bg_img

            overlay = np.zeros_like(bg_img)
            overlay[mask] = color_rgb

            # Use addWeighted for blending: 0.6 original + 0.4 color
            return cv2.addWeighted(bg_img, 0.6, overlay, 0.4, 0)

        # 1. Prepare base image (slightly darkened to highlight colors)
        base_img = (image * 0.8).astype(np.uint8)

        # 2. Define region logic
        # A. Ground Truth (Green): Exclude Ignore region
        mask_gt = (gt > 0) & (gt != 255)

        # B. Prediction (Cyan)
        mask_pred = (pred > 0)

        # C. Overflow (Red): Pred=1 & GT=0 (and GT!=255)
        mask_overflow = (pred > 0) & (gt == 0)

        # D. Missing (Blue): Pred=0 & GT=1 (and GT!=255)
        mask_missing = (pred == 0) & (gt > 0) & (gt != 255)

        # 3. Draw four sub-images
        vis_gt = apply_overlay(base_img.copy(), mask_gt, (0, 255, 0))       # Green
        vis_pred = apply_overlay(base_img.copy(), mask_pred, (0, 255, 255)) # Cyan
        vis_over = apply_overlay(base_img.copy(), mask_overflow, (255, 0, 0)) # Red
        vis_miss = apply_overlay(base_img.copy(), mask_missing, (0, 0, 255))  # Blue

        # 4. Add text labels (White text with Black outline)
        def put_text(img, text):
            loc = (20, 40)
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4) # Outline
            cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2) # Body
            return img

        vis_gt = put_text(vis_gt, "GT (Green)")
        vis_pred = put_text(vis_pred, "Pred (Cyan)")
        vis_over = put_text(vis_over, "Overflow (Red)")
        vis_miss = put_text(vis_miss, "Missing (Blue)")

        # 5. Assemble 2x2 grid
        top_row = np.hstack([vis_gt, vis_pred])
        bot_row = np.hstack([vis_over, vis_miss])
        final_grid = np.vstack([top_row, bot_row])

        # 6. Save (Note: OpenCV uses BGR, need to convert back to BGR here)
        save_path = self._get_save_path(name, "jpg")
        cv2.imwrite(save_path, cv2.cvtColor(final_grid, cv2.COLOR_RGB2BGR))


    def draw_heatmap(self, heatmap_data, tag: str, title: str, colormap_spec=None):
        """
        Draw and save a 2D heatmap.

        This function is responsible for rendering a 2D NumPy array or PyTorch tensor
        into a colored heatmap image and saving it to disk using the internal path
        management system. It specifically supports a custom, non-linear segmented
        colormap to enhance visualization contrast in specific intervals.

        Args:
            heatmap_data (torch.Tensor or np.ndarray): 2D data to visualize.
                The function internally handles conversion to a CPU NumPy array.
            tag (str): Unique identifier name for this visualization, used as part
                       of the filename. e.g., 'Logits_Heatmap_Sample_0'.
            title (str): Image title, typically used to display the sample ID,
                         e.g., 'Sample: 2007_000032'.
            colormap_spec (list, optional): Custom segmented colormap list. If None,
                the default colormap is used. If provided, should be a list of
                [(position, color), ...] color points for building a custom colormap.
        """
        # 1. Input data preprocessing
        if isinstance(heatmap_data, torch.Tensor):
            heatmap_data = heatmap_data.detach().cpu().numpy()

        if heatmap_data.ndim != 2:
            raise ValueError(f"[Visualizer] Error: draw_heatmap received non-2D data (shape: {heatmap_data.shape}).")

        # 2. Custom segmented colormap construction
        # Color points: (position, color)
        # Color values normalized from 0-255 to 0-1
        colors = colormap_spec if colormap_spec else [
            (0.0, (0.0, 0.0, 0.0)),      # 0.0 -> Black
            (0.35, (1.0, 0.0, 0.0)),      # 0.2 -> Red
            (0.65, (0.0, 1.0, 1.0)),      # 0.5 -> Cyan
            (1.0, (1.0, 1.0, 1.0))       # 1.0 -> White
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_grid_map", colors)

        # 3. Image drawing and rendering
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

        # im = ax.imshow(heatmap_data, cmap=custom_cmap, vmin=0.0, vmax=1.0)
        im = ax.imshow(heatmap_data, cmap=custom_cmap, vmin=heatmap_data.min(), vmax=heatmap_data.max())
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # Add colorbar for reference
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        # 4. Image saving
        try:
            save_path = self._get_save_path(tag, 'png')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        except Exception as e:
            print(f"[Visualizer] Error saving heatmap: {e}")
        finally:
            # 5. Resource cleanup (critical step to prevent memory leaks)
            plt.close(fig)


    def draw_mask(self, mask_data, tag: str, title: str):
        """
        Draw and save a semantic segmentation mask using the standard PASCAL VOC palette.

        This function supports input of a single-channel label map (where each pixel value
        is a class index) and maps it to a color image conforming to the VOC specification.
        It is particularly suitable for visualizing GT masks or prediction results.

        Args:
            mask_data (torch.Tensor or np.ndarray): 2D mask to visualize.
                Values should be integer indices (0-255). Device conversion is handled
                automatically for Tensors.
            tag (str): Unique file identifier (e.g., 'GT_Mask').
            title (str): Image title.
        """
        # 1. Input data preprocessing
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.detach().cpu().numpy()

        # Force conversion to 2D and ensure integer dtype for palette indexing
        if mask_data.ndim > 2:
            mask_data = np.squeeze(mask_data)

        if mask_data.ndim != 2:
            print(f"[Visualizer] Warning: draw_mask received invalid shape {mask_data.shape}. Skipping.")
            return

        mask_data = mask_data.astype(np.int32)

        # 2. Palette mapping (Index -> RGB)
        # Result shape is (H, W, 3)
        rgb_mask = VOC_PALETTE[mask_data]

        # 3. Image drawing and rendering
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

        ax.imshow(rgb_mask)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

        # 4. Image saving
        try:
            save_path = self._get_save_path(tag, 'png')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        except Exception as e:
            print(f"[Visualizer] Error saving mask: {e}")
        finally:
            # 5. Resource cleanup
            plt.close(fig)

    def draw_image(self, image_data, tag: str, title: str):
        """
        Save a raw RGB image or a preprocessed image.

        This function supports input of PyTorch tensors or NumPy arrays, and automatically
        handles dimension permutation and numerical normalization. It is specifically used
        for outputting the original image during training/evaluation for spatial alignment
        auditing or error analysis.

        Args:
            image_data (torch.Tensor or np.ndarray): Image data to visualize.
                - If Tensor, shape should be [3, H, W] or [H, W, 3].
                - If ndarray, shape should be [H, W, 3].
            tag (str): Unique file identifier (without suffix).
            title (str): Image title, displayed above the drawing area.
        """
        if isinstance(image_data, torch.Tensor):
            # [3, H, W] -> [H, W, 3]
            image_data = image_data.permute(1, 2, 0).cpu().numpy()

        # If in 0-1 range, convert to 0-255
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.imshow(image_data)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        try:
            save_path = self._get_save_path(tag, 'png')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        finally:
            plt.close(fig)

    def draw_binary_map(self, binary_data, tag: str, title: str):
        """
        Draw and save a binary [0, 1] image.

        This function is specifically designed for visualizing non-continuous, hard-classified
        binary data (e.g., raw NAMLab index masks). It uses an extreme-contrast black-and-white
        colormap, does not display a colorbar, and forces no pixel interpolation to ensure
        that the original spatial structure of binary boundaries is not smoothed.

        Args:
            binary_data (torch.Tensor or np.ndarray): 2D binary data to visualize.
                Values should be strictly within the [0, 1] interval.
            tag (str): Unique file identifier (without suffix).
            title (str): Image title, displayed above the drawing area.
        """
        if isinstance(binary_data, torch.Tensor):
            binary_data = binary_data.detach().cpu().numpy()

        if binary_data.ndim != 2:
            raise ValueError(f"[Visualizer] Error: draw_heatmap received non-2D data (shape: {binary_data.shape}).")

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        # Use gray colormap: 0 is black, 1 is white. Use nearest to keep pixel boundaries sharp.
        ax.imshow(binary_data, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        try:
            save_path = self._get_save_path(tag, 'png')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        finally:
            plt.close(fig)

    def draw_dual_channel_overlay(
        self,
        data_r,
        data_g,
        tag: str,
        title: str,
        label_r: str = "Channel Red",
        label_g: str = "Channel Green"
    ):
        """
        Draw and save a dual-channel red-green overlay comparison image (Overlay Visualization).

        This function maps two 2D tensors to the red and green channels of an RGB image
        respectively. It is primarily used to troubleshoot the alignment of two spatial signals:
        - Red area (Red): Region where only data_r exists.
        - Green area (Green): Region where only data_g exists.
        - Yellow area (Yellow): Region where both overlap heavily (Red + Green = Yellow).

        Algorithm Logic:
        1. Normalize input data to the [0, 1] interval.
        2. Construct an RGB array of shape [H, W, 3], forcing the blue channel to 0.
        3. Use matplotlib's inset_axes or make_axes_locatable to construct two independent Colorbars.

        Args:
            data_r (torch.Tensor or np.ndarray): 2D data mapped to the red channel.
            data_g (torch.Tensor or np.ndarray): 2D data mapped to the green channel.
            tag (str): Unique file identifier (without suffix).
            title (str): Image title.
            label_r (str): Description label for the red channel, used for Colorbar.
            label_g (str): Description label for the green channel, used for Colorbar.
        """
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # 1. Type conversion and dimension check
        if isinstance(data_r, torch.Tensor):
            data_r = data_r.detach().cpu().numpy()
        if isinstance(data_g, torch.Tensor):
            data_g = data_g.detach().cpu().numpy()

        if data_r.shape != data_g.shape:
            raise ValueError(
                f"[Visualizer] Shape mismatch: data_r {data_r.shape} vs data_g {data_g.shape}"
            )
        if data_r.ndim != 2:
            raise ValueError(f"[Visualizer] Input must be 2D, but got {data_r.ndim}D")

        # 2. Data normalization (prevent color distortion from numerical overflow)
        def normalize_visual(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 1e-6:
                return (x - x_min) / (x_max - x_min)
            return np.clip(x, 0, 1)

        norm_r = normalize_visual(data_r)
        norm_g = normalize_visual(data_g)

        # 3. Compose RGB image [H, W, 3]
        h, w = norm_r.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.float32)
        rgb_image[..., 0] = norm_r  # Red
        rgb_image[..., 1] = norm_g  # Green
        # Blue channel is kept at 0, or can be set to a very small value (e.g., 0.1)
        # to show faint background outline

        # 4. Drawing and layout
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        img = ax.imshow(rgb_image, interpolation='nearest')
        ax.set_title(title, fontsize=12, pad=15)
        ax.axis('off')

        # 5. Construct dual independent Colorbars
        # Define custom colormaps: black to pure red, black to pure green
        cmap_r = mcolors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])
        cmap_g = mcolors.LinearSegmentedColormap.from_list("black_green", ["black", "green"])

        divider = make_axes_locatable(ax)

        # Red channel Colorbar (placed on the right)
        ax_cb_r = divider.append_axes("right", size="3%", pad=0.15)
        norm_obj_r = mcolors.Normalize(vmin=data_r.min(), vmax=data_r.max())
        cb_r = fig.colorbar(plt.cm.ScalarMappable(norm=norm_obj_r, cmap=cmap_r), cax=ax_cb_r)
        cb_r.set_label(label_r, fontsize=9)
        cb_r.ax.tick_params(labelsize=8)

        # Green channel Colorbar (placed at the bottom)
        ax_cb_g = divider.append_axes("bottom", size="3%", pad=0.3)
        norm_obj_g = mcolors.Normalize(vmin=data_g.min(), vmax=data_g.max())
        cb_g = fig.colorbar(plt.cm.ScalarMappable(norm=norm_obj_g, cmap=cmap_g), cax=ax_cb_g, orientation='horizontal')
        cb_g.set_label(label_g, fontsize=9)
        cb_g.ax.tick_params(labelsize=8)

        # 6. Save and release
        try:
            save_path = self._get_save_path(tag, 'png')
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        finally:
            plt.close(fig)

    def draw_detections(self, image_data, detections, class_names, tag: str, title: str):
        """
        Draw bounding boxes and class labels on an image and save the result.

        This function supports input of PyTorch tensors or NumPy arrays, and automatically
        handles color space recovery and coordinate rendering. It is designed as a general-purpose
        visualization probe for auditing Grounding DINO's prediction quality or spatial alignment
        after data augmentation.

        Args:
            image_data (torch.Tensor or np.ndarray): Image data to visualize.
                - If Tensor, shape should be [3, H, W] (Float [0, 1]).
                - If ndarray, shape should be [H, W, 3] (Uint8 [0, 255]).
            detections (Detections): Detections object containing prediction info.
                Must support iteration yielding (xyxy, mask, confidence, class_id, tracker_id, data).
            class_names (list): List of class names, used to map class_id to string labels.
            tag (str): Unique file identifier (without suffix), determines the output file path.
            title (str): Image title, displayed above the drawing area.
        """
        # 1. Unified image format conversion and numerical restoration (Normalization Reversal)
        if torch.is_tensor(image_data):
            # Process Tensor: [C, H, W] -> [H, W, C] and convert to [0, 255]
            img_np = image_data.detach().cpu().permute(1, 2, 0).numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # Process Numpy: Ensure Uint8
            img_np = image_data.astype(np.uint8)

        h_img, w_img = img_np.shape[:2]

        # 2. Initialize drawing context (use Agg backend for headless server support)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img_np)
        ax.set_title(title, fontsize=15)
        ax.axis('off')

        # 3. Iterate over Detections object for drawing
        # Internal iterator returns: (xyxy, mask, confidence, class_id, tracker_id, data)
        for i, (xyxy, _, conf, class_id, _, _) in enumerate(detections):
            # Get bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1

            local_name = class_names[class_id] if class_id is not None else "background"

            # Look up the name index in the standard table; default to 0 (background) if not found
            if local_name in standard_voc_labels:
                global_idx = standard_voc_labels.index(local_name)
            else:
                global_idx = 0

            # Use the global index to pick a color, ensuring "person" is always magenta,
            # "bird" is always green, etc.
            color_uint8 = VOC_PALETTE[global_idx]
            color_rgb = [c / 255.0 for c in color_uint8]

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=4,
                edgecolor=color_rgb,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)

            # Build label text: "ClassName Conf"
            name = class_names[class_id] if class_id is not None and class_id < len(class_names) else "Unknown"
            label_text = f"{name} {conf:.2f}" if conf is not None else f"{name}"

            # Draw label background and text
            ax.text(
                x1, y1 - 5, label_text,
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor=color_rgb, edgecolor='none', alpha=0.6, pad=2)
            )

        # 4. Execute path retrieval and saving
        # Depends on the Visualizer's internal path management policy
        # (auto-generates save path based on tag)
        save_path = self._get_save_path(tag, extension="png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)

# =============================================================================
# Global Singleton Instance
# From anywhere, simply `from ... import VISUALIZER` to use.
# =============================================================================
VISUALIZER = Visualizer()