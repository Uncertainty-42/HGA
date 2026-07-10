# model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from sam_lora import LoRA_Sam
from effsam_lora import LoRA_EffSam
from typing import Tuple
from copy import deepcopy



from _Optimization_Workspace.modules.structure_patches import apply_structure_patches as apply_patches_impl

from _Optimization_Workspace.tools.visualize import VISUALIZER, VIRIDIS_SPEC

class Model(nn.Module):
    """
    Co2SAM core model class (Central Processor).

    This class encapsulates the full lifecycle of SAM (Segment Anything Model) and provides
    extended interfaces optimized for the WSSS task.

    Core mechanism: Internally, this model is defined as an "Instance Proposal Generator".
    It does not perform 21-class multi-label semantic segmentation internally; instead,
    for the N prompts provided by an external detector (e.g., DINO), it generates N
    class-agnostic binary masks. Semantic category mapping and competition logic are
    decoupled into downstream loss functions and visualization functions.

    Architecture Overview:
        [Input] -> [ImageEncoder] -> {Feature Hook} -> [MaskDecoder] -> [Output]
                                           ^
                                           | (Zone 2: Feature Modification)

    Extensibility Guide:
        1. Feature-Level Enhancement (Zone 2 - Neck):
           - Mechanism: Use the `self.feature_modifier` interface.
           - Scenario: Introduce NAMLab, Attention, Denoise and other modules for
             intermediate processing of the Feature Map.
           - Method: In `__init__`, instantiate an `nn.Module` based on the config
             and assign it to `self.feature_modifier`.
           - Note: The forward pass automatically handles state synchronization;
             no manual intervention is required.

        2. Encoder Modification (Zone 1 - Encoder):
           - Mechanism: Component Replacement.
           - Scenario: Replace the Backbone (e.g., Swin, ResNet) or inject an Adapter
             inside the Encoder.
           - Method: In `__init__` or `finetune`, directly override
             `self.model.image_encoder`.
             e.g., `self.model.image_encoder = MyCustomEncoder(self.model.image_encoder)`

        3. Decoder Fine-Tuning (Zone 3 - Decoder):
           - Mechanism: Component replacement or post-processing.
           - Scenario: Modify mask generation logic or add CRF post-processing.
           - Method: Override `self.model.mask_decoder` or add processing logic
             before the `forward` return.

    Args:
        cfg (Box): Global configuration object, controlling model structure and
            enabling of extension modules.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None
        self.target_length = 1024
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        # === [Hook] Feature Modifier Interface ===
        self.current_priors = None          # Stores the raw dictionary mounted from the main loop
        self.processed_priors_dict = {}     # Stores energy-band tensors processed according to the Config bill
        self.output_hub = {}                # [Hub] Stores all intermediate outputs that need to be sent to the Loss class for audit
        
        # Reserved for future modules (e.g., NAMLab Neck)
        self.feature_modifier = None


    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_h_4b8939.pth")
        elif model_type == "eff_xl0":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "efficientvit_sam_xl0.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def setup(self):
        """
        Master control for model initialization. Completes SAM instantiation,
        parameter freezing, and LoRA injection.
        """
        model_type = self.cfg.model.type
        model_versions = self.cfg.model.versions
        assert isinstance(model_type, str)
        checkpoint = self.get_checkpoint(model_type)
        if model_type in model_versions["vit"]:
            self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        elif model_type in model_versions["eff_vit"]:
            self.model = create_efficientvit_sam_model(
                name=self.cfg.model.eff_mapping[model_type], 
                pretrained=True,
                weight_url=checkpoint
            )
            print(f"[Info]✅Loaded {self.cfg.model.eff_mapping[model_type]}")

        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.finetune()

    def finetune(self):
        """
        Parameter-Efficient Fine-Tuning (PEFT) configuration.
        Currently responsible for injecting LoRA low-rank matrices.
        """
        model_type = self.cfg.model.type
        model_versions = self.cfg.model.versions
        if model_type in model_versions["vit"]:
            assert isinstance(self.model, Sam)
            LoRA_Sam(self.model, 4)
            print("[Info] ✅ Successfully using original ViT SAM model with LoRA")
        elif model_type in model_versions["eff_vit"]:
            LoRA_EffSam(self.model, self.cfg)
            print("[Info] ✅ Successfully using EfficientViT SAM model with LoRA")


    def apply_structure_patches(self):
        """
        [Architecture Extension] Apply Dynamic Structure Patches.

        This method acts as the bridge connecting Host (model.py) with Plugin
        (_Optimization_Workspace). It reads the `cfg.model.patches` configuration
        and calls processors within the Workspace to perform in-place modifications
        on the model.

        Design Pattern: Proxy / Delegate
        """
        # 1. Get Config
        patches_config = self.cfg.model.get('patches', None)
        # If the config does not exist or is an empty dict, return directly, ensuring zero overhead
        if not patches_config:
            return

        # 3. Delegate Execution
        # Note: We pass the underlying SAM model (self.model), not the Wrapper (self),
        # so that downstream modules can directly access components like mask_decoder
        apply_patches_impl(self.model, patches_config)

    def prepare_for_image(self, priors_payload: dict):
        """
        [Explicit Preprocessing Interface] Image-level prior processing.

        This function is designed to decouple "Prompt-independent" image-level priors
        (such as whole-image depth gradients, NAMLab edge maps) from the inner inference
        loop in the WSSS pipeline. It executes once before processing each raw image,
        completing data transfer, edge extraction, and multi-scale blurring, and persists
        the results in `self.processed_priors_dict` for reuse across all subsequent chunks.

        Args:
            priors_payload (Dict): Raw prior dictionary provided by the DataLoader
                (typically on CPU).
        """
        self.processed_priors_dict.clear()
        
        if priors_payload is None:
            self.current_priors = None
            return

        # 1. Alignment: transfer to the current model's compute device
        device = next(self.parameters()).device
        self.current_priors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in priors_payload.items()
        }

        # 2. Trigger the processing hub: execute costly convolution/gradient ops (only once)
        self._prepare_processed_priors()

    def forward(self, images, prompts, orig_image_size):
        """
        [Main Flow] Execute the complete forward inference logic.

        Args:
            images (torch.Tensor): Input images, shape [B, 3, H, W]. Typically B=1.
            prompts (List[torch.Tensor]): Bounding box prompts, a list of length B,
                each element of shape [N, 4].
            orig_image_size (Tuple[int, int]): Original image resolution (H, W).
        Returns:
            image_embeddings (torch.Tensor): Shape [B, 256, 64, 64], extracted high-level
                image features.
            pred_masks (List[torch.Tensor]): A list of length Batch. Each element is a
                binary instance mask (Logits) of shape [N, 1024, 1024]. Note: N here is
                the number of prompts for this sample in the current chunk, and the channel
                dimension has been removed via squeeze(1).
            ious (List[torch.Tensor]): A list of length Batch. Each element is an instance
                predicted IoU of shape [N, 1].
            res_masks (List[torch.Tensor]): A list of length Batch. Each element is a
                low-resolution mask block of shape [N, 1, 256, 256], preserving original
                sampling dimensions.
        """
        # At this point self.current_priors has been mounted externally; this function
        # processes them into 1024-level edge energy bands

        self.output_hub.clear()             # Clear the shelf before each forward pass
        self._mount_special_loss_inputs()    # Auto-mount layer instances (if needed)

        # 1. Encode
        # Invoke native encoder; self.image_embeddings is assigned the raw features
        image_embeddings = self.encode(images)
        # image_embeddings shape: [B, 256, 64, 64] (image embedding features)
        
        # 2. [Hook] Feature Modification
        # Check whether a modifier (e.g., NAMLab) is mounted
        if self.feature_modifier is not None:
            # Process the features
            image_embeddings = self.feature_modifier(image_embeddings)
            
            # [CRITICAL] State synchronization
            # Must explicitly update self.image_embeddings
            # otherwise subsequent self.decode() would still read the old raw features
            self.image_embeddings = image_embeddings

        # 3. Decode
        H, W = orig_image_size
        pred_masks, ious, res_masks = self.decode((H, W), prompts)
        return image_embeddings, pred_masks, ious, res_masks
    
    def encode(self, images):
        """
        [Encoder Path] Execute image preprocessing and feature extraction.
        """

        # Explicitly invoke preprocessing
        # This automatically performs (x - mean) / std normalization

        # Preprocess first, then image_encoder
        x = self.preprocess(images)
        # x shape: [B, 3, 1024, 1024] (standard input after normalization and padding)
        self.image_embeddings = self.model.image_encoder(x)
        # self.image_embeddings shape: [B, 256, 64, 64] (16x downsampled features extracted by ViT-B)
        return self.image_embeddings 

    def decode(self, image_shape, prompts):
        """
        [Core Decoder] Execute multi-prompt parallel decoding with prior-guided refinement.

        This function is the core logic outlet of Co2SAM. It receives image embeddings
        and a set of prompts, generates preliminary masks through the MaskDecoder, and
        performs gated refinement at 1024 resolution during training.

        Args:
            image_shape (Tuple[int, int]): Target output resolution, (1024, 1024).
            prompts (List[torch.Tensor]): A list of length Batch, each element is a tensor
                of shape [N, 4] bounding box coordinates. (N is the number of prompts for
                the current image)

        Returns:
            pred_masks (List[torch.Tensor]): A list of length Batch, each element is a
                binary logits tensor of shape [N, H, W] (after squeeze). (N is the number
                of prompts for the current image)
            ious (List[torch.Tensor]): A list of length Batch, each element is an instance
                confidence score of shape [N, 1].
            res_masks (List[torch.Tensor]): A list of length Batch, each element is a
                low-resolution raw logits tensor of shape [N, 1, 256, 256].
        """
        image_embeddings = self.image_embeddings
        if image_embeddings == None:
            raise ValueError("No image embeddings")

        pred_masks = []
        ious = []
        res_masks = []

        for prompt, embedding in zip(prompts, image_embeddings):
            # embedding shape: [256, 64, 64] (unpacked from the previous single sample level)

            if isinstance(prompt, torch.Tensor):
                # in this way

                prompt = prompt.to(device=embedding.device)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=prompt,
                masks=None,
            )
            elif isinstance(prompt, tuple):

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=prompt,
                boxes=None,
                masks=None,
            )
            # sparse_embeddings = [2, 2, 256], minmax = [-1.38, 1.71]
            # dense_embeddings = [2, 256, 64, 64], minmax = [-0.18, 0.39]


            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # low_res_masks shape: [N, 1, 256, 256] (N is the number of prompts for the current image)
            # iou_predictions shape: [N, 1] (confidence score for each instance)
            

            # No distinction between training and validation, since sizes are all 1024
            masks = F.interpolate(
                low_res_masks,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )
            # masks shape: [N, 1, 1024, 1024] (upsampled via bilinear interpolation, channel count stays 1)
                
            # ===Centralized Hub data warehousing ===
            if self.cfg.model.get('patches', {}).get('mask_decoder_upscaling') == 'prior_guided_upscaling':
                self.output_hub.setdefault('logits_prior_guided_upscaling_256', []).append(low_res_masks)
            
            if self.cfg.model.get('post_decoder', {}).get('decoder_output_refinement') == 'boundary_gating':
                self.output_hub.setdefault('logits_boundary_gating_1024', []).append(masks)
            # ===================================

            if len(self.cfg.loss.weights.get('top2_edge_alignment', {})) > 0:
                self.output_hub.setdefault('logits_256', []).append(low_res_masks)
                self.output_hub.setdefault('logits_1024', []).append(masks)

            # print(f"[Debug]🚗🚗🚗 In model.py, Model.decode: output_hub: {self.output_hub}")

            pred_masks.append(masks.squeeze(1))
            # Here masks have shape [N, 1, 1024, 1024].
            # The tensor returned to the main loop is [N, 1024, 1024] to support subsequent
            # Top1-Top2 competition loss computation.
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)

        return pred_masks, ious, res_masks
    
    @staticmethod
    def get_preprocess_shape1(oldh, oldw, long_side_length):
        """
        Compute the target dimensions after image scaling while preserving aspect ratio.

        Args:
            oldh (int): Original height.
            oldw (int): Original width.
            long_side_length (int): Target length for the long side after scaling
                (typically 1024).

        Returns:
            tuple: Scaled target dimensions (new_h, new_w).
        """

        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords_torch1(self, coords, original_size):
        """
        Map pixel coordinates from the original image space to SAM's 1024x1024
        center-aligned space.

        Steps:
        1. Compute the long-side scale factor.
        2. Apply proportional coordinate scaling.
        3. Add the offset induced by center padding to complete coordinate alignment.

        Args:
            coords (torch.Tensor): Original coordinate tensor.
            original_size (tuple): Original image dimensions (H, W).

        Returns:
            torch.Tensor: Mapped coordinates in 1024 space.
        """
        old_h, old_w = original_size

        # breakpoint()
        oldh1 = int(original_size[0])
        new_h, new_w = self.get_preprocess_shape1(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        # Compensate for displacement caused by center padding
        max_dim = self.target_length
        pad_h = (max_dim - new_h) // 2
        pad_w = (max_dim - new_w) // 2
        coords[..., 0] += pad_w
        coords[..., 1] += pad_h

        return coords

        
    def apply_boxes_torch1(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ):
        """
        Map bounding boxes from the original image space to SAM's 1024x1024
        center-aligned space.

        Args:
            boxes (torch.Tensor): Original bounding boxes of shape [N, 4] (xyxy).
            original_size (tuple): Original image dimensions (H, W).

        Returns:
            torch.Tensor: Mapped bounding boxes of shape [N, 4].
        """

        boxes = self.apply_coords_torch1(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Image preprocessing operator: performs normalization and center padding.

        Steps:
        1. Apply color space normalization using preset mean and standard deviation.
        2. Compute the size differences between the 1024 standard canvas and the
           current image.
        3. Apply center padding to ensure that effective image content is positioned
           at the very center of the canvas.

        Args:
            x (torch.Tensor): Input image tensor, shape [B, 3, h, w].

        Returns:
            torch.Tensor: Padded standard input, shape [B, 3, 1024, 1024].
        """
        # Step 1: Per-pixel color space normalization (subtract mean / divide by std)
        # Ensure pixel_mean/pixel_std reside on the same device as the input image
        x = (x - self.pixel_mean.cuda()) / self.pixel_std.cuda()

        # Step 2: Compute center padding geometric parameters
        h, w = x.shape[-2:]
        # Must follow the same pattern as the training pipeline: use center, not top-left
        max_dim = self.cfg.model_img_size or 1024
        padh_total = max_dim - h
        padw_total = max_dim - w

        # Distribute total padding evenly to both sides to achieve centering
        pad_top = padh_total // 2
        pad_left = padw_total // 2
        
        # Step 3: Apply padding
        # F.pad argument order: (left, right, top, bottom)
        x = F.pad(x, (pad_left, padw_total - pad_left, pad_top, padh_total - pad_top))
        return x
    
    def _mount_special_loss_inputs(self):
        """[Internal] Auto-mount layer instances required by specific losses
        based on weight configuration (decoupling logic)."""
        w = self.cfg.loss.weights
        if w.get('grid_penalty', 0) > 0 or w.get('scvc_penalty', 0) > 0:
            # Only store layer instance in hub when the relevant loss is active
            self.output_hub['grid_loss_inputs'] = {
                'conv_t_layers': [
                    self.model.mask_decoder.output_upscaling[0],
                    self.model.mask_decoder.output_upscaling[3]
                ]
            }

    # =======================================================================
    # Hierarchical-Geometric Priors Processing Hub
    # =======================================================================
    def _prepare_processed_priors(self):
        """
        Prior data processing hub.
        Based on processed_requests (parameter tuple bills) in cfg.priors,
        performs edge extraction, gradient computation, and Gaussian blurring
        on the raw NAMLab index map and Depth depth map, and stores the results
        in a dictionary keyed by "unique string concatenated from parameter tuples",
        enabling GPU compute reuse and deduplication.
        """
        self.processed_priors_dict.clear()
        
        if not hasattr(self, 'current_priors') or self.current_priors is None:
            return

        # ---------------------------------------------------------
        # 1. Process NAMLab (discrete region map -> continuous boundary energy band)
        # ---------------------------------------------------------
        namlab_raw = self.current_priors.get('namlab')
        # namlab_raw shape: [B, 1, 1024, 1024] (raw region indices, discrete values)
        namlab_reqs = getattr(self.cfg.get("priors", {}).get("namlab", {}), 'processed_requests', set())
        
        if namlab_raw is not None and namlab_reqs:
            # print(f"[Warning] ⚠️ In model.py namlab_raw {namlab_raw is not None}, {self.cfg.priors.namlab.processed_requests}")
            # Ensure channel dimension [N, 1, H, W]
            if namlab_raw.ndim == 3:
                namlab_raw = namlab_raw.unsqueeze(1)
            
            # Step 1: Extract 0/1 binary discrete boundaries (compute only once)
            namlab_edges = self._extract_discrete_boundaries(namlab_raw)
            # namlab_edges shape: [B, 1, 1024, 1024] (binarized boundary, 0 or 1)

            # Step 2: Generate energy bands at different blur levels according to the bill
            for req_tuple in namlab_reqs:
                params = dict(req_tuple)
                sigma = params.get('gaussian_sigma', 1.0)
                
                tensor_blurred = self._apply_gaussian_blur(namlab_edges, sigma)
                # tensor_blurred shape: [B, 1, 1024, 1024] (continuous energy map after Gaussian blur)
                
                key_name = "namlab_" + "_".join([f"{k}_{v}" for k, v in req_tuple])
                self.processed_priors_dict[key_name] = tensor_blurred
                # print(f"[Info] model.processed_priors_dict.keys(): {self.processed_priors_dict.keys()}")

        # ---------------------------------------------------------
        # 2. Process Depth (continuous depth map -> gradient boundary energy band)
        # ---------------------------------------------------------
        depth_raw = self.current_priors.get('depth')
        # depth_raw shape: [B, 1, 1024, 1024] (raw depth generated by depth-anything, continuous values)
        depth_reqs = getattr(self.cfg.get("priors", {}).get("depth", {}), 'processed_requests', set())
        
        if depth_raw is not None and depth_reqs:
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.unsqueeze(1)

            for req_tuple in depth_reqs:
                params = dict(req_tuple)
                d_tensor = depth_raw.clone() # Deep clone to prevent contamination across multiple bill parameters
                
                # Step A: Per-image normalization
                if params.get('apply_norm', False):
                    d_min, d_max = d_tensor.min(), d_tensor.max()
                    if d_max - d_min > 1e-6:
                        d_tensor = (d_tensor - d_min) / (d_max - d_min)
                    else:
                        d_tensor = d_tensor - d_min
                
                # Step B: Gradient operator
                grad_method = params.get('gradient_method', 'none')
                if grad_method == 'sobel':
                    d_tensor = self._compute_sobel_gradient(d_tensor)

                if VISUALIZER.is_active("depth_prior_generation"):
                    VISUALIZER.draw_heatmap(d_tensor.squeeze(), tag="depth_grad", title="depth_grad", colormap_spec=VIRIDIS_SPEC)
                # Step C: Hill equation distribution shift (Hill Shift) ---
                # Parameter tuple format: (x_center, y_center, slope_k)
                shift_params = params.get('hill_shift', None)
                if shift_params is not None:
                    xc, yc, k = shift_params
                    eps = 1e-8
                    # Compute shift coefficient M so that the curve passes precisely through (xc, yc)
                    M = (xc / (1.0 - xc + eps)) * (((1.0 - yc) / (yc + eps))**(1.0 / k))
                    
                    # Hill equation formula: y = x^k / (x^k + [M(1-x)]^k)
                    x_k = d_tensor.pow(k)
                    denom_part = (M * (1.0 - d_tensor + eps)).pow(k)
                    d_tensor = x_k / (x_k + denom_part + eps)

                if VISUALIZER.is_active("depth_prior_generation"):
                    VISUALIZER.draw_heatmap(d_tensor.squeeze(), tag="depth_grad_hill_processed", title="depth_grad_hill_processed", colormap_spec=VIRIDIS_SPEC)

                # Step D: Hard threshold binarization ---
                thresh = params.get('binarize_threshold', None)
                if thresh is not None:
                    d_tensor = (d_tensor >= thresh).float()
                
                # Step E: Gaussian blurring
                sigma = params.get('gaussian_sigma', 1.0)
                d_tensor = self._apply_gaussian_blur(d_tensor, sigma)
                
                key_name = "depth_" + "_".join([f"{k}_{v}" for k, v in req_tuple])
                self.processed_priors_dict[key_name] = d_tensor
                # print(f"[Info] model.processed_priors_dict.keys(): {self.processed_priors_dict.keys()}")

    # ------------------ [Internal Math Operator Engines] ------------------

    def _extract_discrete_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Ultra-fast algorithm: find index differences via a 3x3 pooling window
        to extract discrete boundaries."""
        x = mask.float()
        # Maximum within the 3x3 window
        max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # Minimum within the 3x3 window (implemented via -max(-x))
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        # Where max differs from min is the boundary
        boundary = (max_pool != min_pool).float()
        return boundary

    def _compute_sobel_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Use a 3x3 Sobel operator to extract abrupt structural changes in
        continuous space and forcibly normalize the energy."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Strictly clamp gradient energy to [0, 1] to prevent downstream loss imbalance
        g_min, g_max = grad_mag.min(), grad_mag.max()
        if g_max - g_min > 1e-6:
            grad_mag = (grad_mag - g_min) / (g_max - g_min)
        else:
            grad_mag = grad_mag - g_min
            
        return grad_mag

    def _apply_gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Dynamically generate and apply a 2D Gaussian convolution kernel
        according to Sigma."""
        if sigma <= 0:
            return x
            
        # Set kernel size according to the empirical rule (8*sigma) and ensure it is odd
        k_size = int(8 * sigma + 0.5)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        k_size = max(3, k_size)
            
        # Dynamically generate a 1D Gaussian kernel
        coords = torch.arange(k_size, dtype=torch.float32, device=x.device)
        coords -= (k_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g_1d /= g_1d.sum()

        g_h = g_1d.view(1, 1, 1, k_size)
        g_v = g_1d.view(1, 1, k_size, 1)
    
        # 3. Apply separable convolution
        blurred = F.conv2d(x, g_h, padding=(0, k_size // 2))
        blurred = F.conv2d(blurred, g_v, padding=(k_size // 2, 0))
        
        # 4. [Key Point] Apply scale compensation coefficient
        # Multiply by sigma^4 so that as sigma increases, the full-field prior intensity
        # and gradient magnitude are enhanced synchronously, and points farther from
        # the origin experience more pronounced enhancement
        if VISUALIZER.is_active("show_blurred_without_sigma^4_Comp"):
            VISUALIZER.draw_heatmap(blurred.squeeze(), tag="blurred without sigma^4 Comp.", title="blurred without $\sigma^4$ Comp.", colormap_spec=VIRIDIS_SPEC)
        return blurred * (sigma ** 4)