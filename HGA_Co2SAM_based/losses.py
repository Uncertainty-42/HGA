# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from _Optimization_Workspace.tools.visualize import VISUALIZER, VIRIDIS_SPEC

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1, valid_mask=None):
        """
        Args:
            valid_mask (torch.Tensor, optional): Geometric validity mask [1, 1024, 1024].
        """
        inputs = F.sigmoid(inputs)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        if valid_mask is not None:
            BCE = BCE * valid_mask
            # Corrected mean: denominator only counts valid pixels
            BCE_mean = BCE.sum() / (valid_mask.sum() * inputs.size(0)).clamp(min=1.0)
        else:
            BCE_mean = BCE.mean()

        BCE_EXP = torch.exp(-BCE_mean)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE_mean

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1, valid_mask=None):
        inputs = F.sigmoid(inputs)

        # Geometric masking: compute intersection and union only within the valid region
        if valid_mask is not None:
            inputs = inputs * valid_mask
            targets = targets * valid_mask
            intersection = (inputs * targets).sum()
            # Denominator is also computed within the valid region
            denominator = inputs.sum() + targets.sum()
        else:
            intersection = (inputs * targets).sum()
            denominator = inputs.sum() + targets.sum()

        dice = (2. * intersection + smooth) / (denominator + smooth)

        return 1 - dice


class ContraLoss(nn.Module):

    def __init__(self, temperature = 0.3, weight=None, size_average=True):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedd_x: torch.Tensor, embedd_y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor, valid_mask=None):
        """
        Args:
            valid_mask (torch.Tensor, optional): Valid region mask of 1024x1024.
        """
        x_embedding = self.norm_embed(embedd_x) # embedd_x: [256, 64, 64]
        y_embedding = self.norm_embed(embedd_y)

        # 1. Synchronously downsample the validity mask to 64x64 (must use nearest-neighbor mode)
        if valid_mask is not None:
            v_mask_64 = F.interpolate(valid_mask, size=x_embedding.shape[-2:], 
                                      mode="nearest").detach()
        else:
            v_mask_64 = 1.0

        # 2. Correct mask weights after interpolation: remove edge noise introduced by interpolation via v_mask_64
        x_masks = F.interpolate(mask_x, size=x_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        x_masks = x_masks * v_mask_64 # Geometric masking

        # 3. Correct the denominator sum_x: ensure only the weight sum within the valid region is computed
        sum_x = x_masks.sum(dim=[-1, -2]).clone()

        # Apply masking on the y branch identically
        y_masks = F.interpolate(mask_y, size=y_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        y_masks = y_masks * v_mask_64
        sum_y = y_masks.sum(dim=[-1, -2]).clone()

        multi_embedd_x = (x_embedding * x_masks).sum(dim=[-1, -2]) / sum_x.clamp(min=1e-6)  # [n, 256, 64, 64] >> [n, 256]
        multi_embedd_y = (y_embedding * y_masks).sum(dim=[-1, -2]) / sum_y.clamp(min=1e-6)

        flatten_x = multi_embedd_x.view(multi_embedd_x.size(0), -1)
        flatten_y = multi_embedd_y.view(multi_embedd_y.size(0), -1)

        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_y.unsqueeze(0), dim=2)

        # 1. Apply temperature coefficient
        similarity_matrix = similarity_matrix / self.temperature
        # 2. Construct target labels: the diagonal represents positive pair indices (0, 1, 2, ..., N-1)
        targets = torch.arange(similarity_matrix.size(0), device=embedd_x.device)
        # 3. Use built-in cross-entropy; its internal Log-Sum-Exp mechanism prevents exponential explosion
        loss = self.criterion(similarity_matrix, targets)

        return loss

    def norm_embed(self, embedding: torch.Tensor):
        embedding = F.normalize(embedding, dim=0, p=2)
        return embedding


class Top2EdgeAlignmentLoss(nn.Module):
    """
    Decision-Competition edge alignment loss based on Top1-Top2 margin sampling
    (Boundary Alignment via Logit Competition).

    The core logic of this loss is to leverage the "uncertainty zone" in multi-class
    decision-making to model the Semantic Edge. When the model exhibits intense competition
    between two categories (Top1 probability close to Top2 probability), that pixel is
    considered to lie on a semantic boundary. This module forces this "competition zone"
    to spatially align with the Hierarchical-Geometric Priors (NAMLab/Depth) via attractor-based loss.

    Computation:
        1. Prob = Softmax(Logits, dim=channel)
        2. P1, P2 = TopK(Prob, k=2)
        3. Semantic_Edge = 1.0 - (P1 - P2)
        4. Loss = weighted_overlap(Semantic_Edge * Valid_Mask, Prior_Attractor_Field * Valid_Mask)

    Design principles:
        - Zero-parameter: Fully based on decision probability distribution, introducing no extra conv layers.
        - Isolated: Invalid padding regions from SAM are excluded via valid_mask.
        - Scale-agnostic: Input resolution is controlled externally; this class only handles
          pixel-level alignment computation.
    """

    def __init__(self, eps=1e-6):
        """
        Initialize Top2EdgeAlignmentLoss.

        Args:
            eps (float): Smoothing constant to prevent division by zero.
        """
        super().__init__()
        self.eps = eps

    def _get_gradient_magnitude(self, x):
        """Compute the gradient magnitude of an image tensor (Sobel operator)."""
        # Define Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        # Compute gradients in both directions
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        # Return magnitude
        return torch.sqrt(gx**2 + gy**2 + self.eps)

    def forward(self, semantic_edge, target_edge, valid_mask=None):
        """
        Compute the edge alignment loss.

        Args:
            semantic_edge (torch.Tensor): Pre-computed Semantic Edge Probe (Top1-Top2
                confidence map), shape [N, 1, H, W].
            target_edge (torch.Tensor): Prior-Guided Attractor Field energy band
                (after Gaussian blurring), shape [1, 1, H, W] or [N, 1, H, W].
            valid_mask (torch.Tensor, optional): Geometric validity mask,
                shape [1, 1, H, W] or [N, 1, H, W].
                Used to mask gradients in SAM padding regions.

        Returns:
            torch.Tensor: Average edge alignment loss per prompt.
        """
        # --- [Loss Internal Audit] Visualize Top1-Top2 boundary confidence ---
        if VISUALIZER.is_active('top2_edge_alignment_monitoring'):

            # Red-green overlay inspection
            # Note: use squeeze() to ensure 2D [H, W] tensors are fed to the plotting function
            VISUALIZER.draw_dual_channel_overlay(
                data_r=target_edge.squeeze(),      # Red channel: Prior-Guided Attractor Field (Target)
                data_g=semantic_edge.squeeze(),    # Green channel: Predicted Semantic Edge Probe (Pred)
                tag="Internal_02_Edge_Overlay", 
                title="Red(Target) vs Green(Pred) Alignment",
                label_r="Prior Attractor (NAMLab/Depth)",
                label_g="Semantic Competition (Top1-Top2)"
            )

        # 1. Spatial geometric masking
        # If a validity mask is provided, exclude padding regions to prevent spurious alignment gradients
        if valid_mask is not None:
            # Ensure target_edge is broadcast-aligned with valid_mask
            # Note: target_edge is expected to be continuous values in [0, 1] (energy band)
            semantic_edge = semantic_edge * valid_mask
            target_edge = target_edge * valid_mask

        # 2. Compute the attractor loss over the prior field (Mountain-based Attraction)
        # Use the blurred original map (mountain shape) directly, rather than gradients (bimodal shape)
        with torch.no_grad():
            # The blur function internally already includes sigma^4 compensation; no normalization needed
            prior_field = target_edge.detach()

            # Obtain the peak altitude of the current prior field
            peak_val = prior_field.max()
            # Construct the valley: 0 at the peak, increasing penalty away from it
            # Distant regions have near-zero gradient, satisfying the "ignore distant regions" requirement
            penalty_map = peak_val - prior_field
        
        # Core computation: maximize the overlap between predicted edges and the prior peak
        # Using negation to convert "maximize overlap" into "minimize loss"
        effective_mask = valid_mask if valid_mask is not None else 1.0
        weighted_overlap = (semantic_edge * penalty_map * effective_mask).sum(dim=(1, 2, 3))
        
        # Retain this normalization: prevents the model from escaping the loss by predicting nothing
        # It normalizes prediction intensity, not prior intensity
        pred_edge_sum = (semantic_edge * effective_mask).sum(dim=(1, 2, 3))
        
        # Final loss: closer to the peak (higher prior_field) → smaller loss (more negative)
        loss = weighted_overlap / (pred_edge_sum + self.eps)

        return loss.mean()

class BoundaryThinningLoss(nn.Module):
    """
    Boundary-Thinning Loss (Boundary Thinning Loss).

    This loss function aims to suppress the bloating of the Semantic Edge Probe
    (Top1-Top2). By monitoring the "total uncertainty energy" within a local region,
    when the energy exceeds a preset budget (Threshold), a Hinge Loss penalty is imposed,
    forcing the model to make decisive boundary calls and shrink "areal" edges back to
    "linear" form.

    Computation:
        1. Local_Energy = P * Kernel_5x5 (where P is the Semantic Edge Probe and Kernel is an all-ones conv kernel)
        2. Loss = mean(max(0, Local_Energy - Threshold))

    Design principles:
        - Local budget mechanism: Allows normal thin edges to exist (does not trigger penalty),
          only suppressing diffuse decision zones.
        - Geometry-sensitive: Cooperates with valid_mask to exclude invalid regions,
          ensuring penalties only occur within the actual image.
    """

    def __init__(self):
        super().__init__()
        # Initialize a 5x5 all-ones convolution kernel for computing local energy sums
        self.register_buffer('kernel', torch.ones((1, 1, 5, 5)))

    def forward(self, semantic_edge, threshold=10.0, valid_mask=None):
        """
        Compute the boundary thinning loss.

        Args:
            semantic_edge (torch.Tensor): Semantic Edge Probe [N, 1, H, W].
            threshold (float): Upper bound of the local energy budget.
            valid_mask (torch.Tensor, optional): Geometric validity mask.

        Returns:
            torch.Tensor: Average per-pixel thinning penalty.
        """
        
        if VISUALIZER.is_active("boundary_thinning_monitoring"):
            # semantic_edge is [1, 1, H, W]; squeeze to 2D for draw_heatmap
            VISUALIZER.draw_heatmap(
                semantic_edge.squeeze(), 
                tag="Internal_01_Top2_Confidence", 
                title="top1-top2_confidence", 
                colormap_spec=VIRIDIS_SPEC
            )
            
        # 1. Spatial geometric masking
        if valid_mask is not None:
            semantic_edge = semantic_edge * valid_mask

        # 2. Local energy monitoring: compute total confidence within a 5x5 window
        # Use F.conv2d for summation; padding=2 preserves spatial resolution
        local_energy = F.conv2d(semantic_edge, self.kernel.to(semantic_edge.device, dtype=semantic_edge.dtype), padding=2) # type: ignore

        # 3. Suppression computation: Hinge Loss penalizes overspending
        loss = F.relu(local_energy - threshold)

        # 4. Normalize and return
        if valid_mask is not None:
            return loss[valid_mask.bool()].mean()
        return loss.mean()


class Co2SAMCriterion(nn.Module):
    """
    Co2SAM unified loss computation arbiter (Facade Pattern).

    This class plays the role of an "arbiter", encapsulating all independent loss
    computations (e.g., Focal, Dice, Contrastive) behind a single unified interface.
    The training loop only needs to feed the model outputs to this class to obtain
    a dictionary containing the total loss and all component losses.

    The core advantages of this design pattern are:
    1.  **Decoupling:** The training loop is separated from concrete loss computation logic.
    2.  **Extensibility:** Adding new losses only requires modifying this
        class internally, without altering the main training logic.
    3.  **Configuration-Driven:** All loss weights and switches are centrally managed
        by the `cfg` object.
    """
    def __init__(self, cfg):
        """Initialize Co2SAMCriterion.

        Based on the provided configuration object `cfg`, this constructor instantiates
        all required low-level loss functions (e.g., FocalLoss, DiceLoss) and sets their
        weights. For optional losses like GridLoss, strict checks are performed against
        the configuration before deciding whether to enable them.

        Args:
            cfg (Box): A Box object containing all experiment configurations, in particular
                `cfg.loss.weights` is required.
        """
        super().__init__()
        self.cfg = cfg
        self.weights = cfg.loss.weights
        if cfg.dataset == "PascalVOC":
            self.channels = 21
        elif cfg.dataset == "COCO":
            self.channels = 81
        else: raise ValueError("❌[in losses.py] Undefined dataset!")
        for weight, value in self.weights.items():
            if weight in ["top2_edge_alignment", "boundary_thinning"]:
                for w, v in value.items():
                    if not isinstance(v, (int, float)) or v < 0:
                        raise ValueError(f"[ERROR] Invalid loss coefficient: {weight}.{w}: {v}, must be non-negative.")
            elif not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"[ERROR] Invalid loss coefficient: {weight}: {value}, must be non-negative.")
        # Reuse already-defined Loss classes from this file
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.contra_loss = ContraLoss()

        # [New Loss 1] Conditionally initialize Top2EdgeAlignmentLoss based on configuration
        self.top2_edge_loss = None
        if 'top2_edge_alignment' in self.weights and self.weights['top2_edge_alignment'] is not None:
            top2_cfg = self.weights['top2_edge_alignment']
            
            # 1. Strict type check
            if not isinstance(top2_cfg, dict):
                raise TypeError(
                    f"top2_edge_alignment weight must be a dict, but received: {type(top2_cfg)}"
                )
            # 2. Strict sub-item numeric range check
            for key, value in top2_cfg.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"top2_edge_alignment sub-weight '{key}' must be non-negative, but received: {value}"
                    )
            if any(v > 0 for v in top2_cfg.values()):
                self.top2_edge_loss = Top2EdgeAlignmentLoss()
                print("[Info] ✅ Top2EdgeAlignmentLoss enabled")

        # [New Loss 2] Conditionally initialize BoundaryThinningLoss based on configuration
        self.boundary_thinning_loss = None
        if 'boundary_thinning' in self.weights and self.weights['boundary_thinning'] is not None:
            thin_cfg = self.weights['boundary_thinning']
            
            # 1. Strict type check
            if not isinstance(thin_cfg, dict):
                raise TypeError(
                    f"boundary_thinning weight must be a dict, but received: {type(thin_cfg)}"
                )
            # 2. Key parameter existence check
            if 'threshold' not in thin_cfg:
                raise KeyError("boundary_thinning config is missing the required 'threshold' key.")

            # 3. Strict sub-item numeric range check (reusing the generic error format from top of __init__)
            for key, value in thin_cfg.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"boundary_thinning sub-weight '{key}' must be non-negative, but received: {value}"
                    )
            
            self.boundary_thinning_loss = BoundaryThinningLoss()
            print("[Info] ✅ BoundaryThinningLoss enabled")

    def forward(self, pred_mask, soft_mask, template_mask, 
                soft_embed, temp_embed, soft_res, temp_res, 
                total_num_masks, output_hub=None, priors_payload=None, 
                processed_priors=None, chunk_voc_ids=None, epoch=None, iter=None,image_path=None):
        """Compute the sum and component breakdown of all losses.

        Args:
            pred_mask (torch.Tensor): [N, 1024, 1024] raw logit masks output by the student network.
                N is the number of category instances in this sample. Same below.
            soft_mask (torch.Tensor): [N, 1024, 1024] pseudo-label masks from U-Teacher (already Sigmoid-ed).
            template_mask (torch.Tensor): [N, 1024, 1024] masks output by the template matching branch.
            soft_embed (torch.Tensor): [256, 64, 64] image embedding from U-Teacher.
            temp_embed (torch.Tensor): [256, 64, 64] image embedding from the template matching branch.
            soft_res (torch.Tensor): [N, 1, 256, 256] low-resolution masks from U-Teacher.
            temp_res (torch.Tensor): [N, 1, 256, 256] low-resolution masks from the template matching branch.
            total_num_masks (int): Total number of prompts in the current batch.
            output_hub (dict, optional): Hub of model internal intermediates.
                Contains intermediate logits or layer references produced by the model,
                such as 'grid_loss_inputs', 'logits_prior_guided_upscaling_256',
                'logits_boundary_gating_1024', etc.
                Example tensor shapes:  ('logits_256', torch.Size([N, 1, 256, 256])),
                                        ('logits_1024', torch.Size([N, 1, 1024, 1024]))
            priors_payload (Dict, optional): Prior payload dictionary containing valid_mask.
                Example tensor shapes:  ('valid_mask', torch.Size([1, 1024, 1024])),
                                        ('namlab', torch.Size([1, 1024, 1024])),
                                        ('depth', torch.Size([1, 1024, 1024]))
            processed_priors (dict, optional): Prior energy bands after model processing.
                Contains tensors with gradients produced by model operators
                (e.g., 'namlab_gaussian_sigma_1.5'). Keys are dynamically determined by
                the internal bill registration logic of the model.
                Example tensor shapes:  ('namlab_gaussian_sigma_3.5', torch.Size([1, 1, 1024, 1024])),
                                        ('depth_apply_norm_True_gradient_method_sobel_hill_shift_(0.2, 0.7, 1.0)_gaussian_sigma_3.5', torch.Size([1, 1, 1024, 1024]))
            chunk_voc_ids (torch.Tensor, optional): [N] list of VOC category IDs corresponding to the current chunk.
            epoch (Optional[int]): Epoch value used for dynamic loss coefficient strategy switching.
            iter (Optional[int]): Iteration value used for dynamic loss coefficient strategy switching.

        Returns:
            dict: A dictionary containing all loss values, with at least the 'total' key.
        """
        dynamic_loss_switch = self.cfg.opt.get("dynamic_loss_switch", None)
        if dynamic_loss_switch is not None and epoch is not None and iter is not None:
            for switch in dynamic_loss_switch:
                if epoch < switch["epoch"]  or (epoch == switch["epoch"] and iter <= switch["iter"]) or switch["epoch"] == -1:
                    s_focal, s_dice, s_template, s_contra, s_align, s_thin = switch["switch"]
                    break
        else:
            s_focal, s_dice, s_template, s_contra, s_align, s_thin = (1, 1, 1, 1, 1, 1)
        # 1. Preprocessing (keep consistent with original logic)
        # Binarize Template Mask
        template_mask_bin = (template_mask > 0.).float()
        # Binarize Soft Mask (for Focal/Dice)
        soft_mask_bin = (soft_mask > 0.).float()

        # 2. Extract validity mask
        valid_mask = None
        if priors_payload is not None:
            valid_mask = priors_payload.get('valid_mask')
            # [Dimension alignment] Ensure the validity mask conforms to PyTorch 4D operator contract [B, 1, H, W]
            if valid_mask is not None and valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)

        # 3. Compute component losses
        # Contrast: note that original logic passes soft_res.clone().detach()
        loss_contra = self.contra_loss(soft_embed, temp_embed, soft_res, temp_res, valid_mask=valid_mask) if self.weights.get('contrast', 0.) > 0. else 0.
        
        # Template: 0.5 Pred + 0.5 Soft
        # Note: original logic has pred_mask as logits (not Sigmoid-ed); dice_loss internally applies Sigmoid
        loss_template = (0.5 * self.dice_loss(pred_mask, template_mask_bin, valid_mask=valid_mask) + 
                         0.5 * self.dice_loss(soft_mask, template_mask_bin, valid_mask=valid_mask)) if self.weights.get('template', 0.) > 0. else 0.
        
        # Focal & Dice
        loss_focal = self.focal_loss(pred_mask, soft_mask_bin, valid_mask=valid_mask) if self.weights.get('focal', 0.) > 0. else 0.
        loss_dice = self.dice_loss(pred_mask, soft_mask_bin, valid_mask=valid_mask) if self.weights.get('dice', 0.) > 0. else 0.

        # 4. Weighted summation
        total_loss = (s_focal * self.weights.get('focal', 0.) * loss_focal +
                      s_dice * self.weights.get('dice', 0.) * loss_dice +
                      s_template * self.weights.get('template', 0.) * loss_template +
                      s_contra * self.weights.get('contrast', 0.) * loss_contra)

        loss_dict = {
            "total": total_loss,
            "focal": s_focal * loss_focal,
            "dice": s_dice * loss_dice,
            "template": s_template * loss_template,
            "contrast": s_contra * loss_contra
        }

        top2_edge_probes = {} # Edge probe cache shared between alignment and thinning losses
        # --- [New Loss 1] Compute and merge Top2EdgeAlignmentLoss (if enabled) ---
        if self.top2_edge_loss and chunk_voc_ids is not None:
            # 1. Contract addressing: read the agreed processing parameter scheme from the prior_configs bill
            bill = self.cfg.model.prior_configs.loss.top2_edge_alignment
            # 2. Precise retrieval: construct key names according to the model-side concatenation logic and extract tensors from the warehouse
            nam_edge = processed_priors.get("namlab_" + "_".join([f"{k}_{v}" for k, v in bill.namlab])) if processed_priors else None
            dep_edge = processed_priors.get("depth_" + "_".join([f"{k}_{v}" for k, v in bill.depth])) if processed_priors else None

            assert nam_edge is not None or dep_edge is not None, "[Error] In losses.py, criterion.forward, top2_edge_loss: nam_edge and depth cannot both be None!"

            # [Dimension alignment] Ensure edge energy bands conform to 4D operation requirements
            if nam_edge is not None and nam_edge.ndim == 3:
                nam_edge = nam_edge.unsqueeze(1)
            if dep_edge is not None and dep_edge.ndim == 3:
                dep_edge = dep_edge.unsqueeze(1)
            
            top2_cfg = self.weights.get('top2_edge_alignment', {})

            assert output_hub, "[Error] In losses.py Co2SAMCriterion forward: output_hub cannot be None!"

            # 2. Perform semantic aggregation: convert list to num_classes-channel tensor
            # Aggregation at 1024 level
            if output_hub.get('logits_1024') is not None:
                # Lazy-load 1024 probe
                if 'logits_1024' not in top2_edge_probes and output_hub.get('logits_1024') is not None:
                    logits_classes_ch_1024 = self._assemble_semantic_logits(
                        output_hub['logits_1024'], chunk_voc_ids, resolution=(1024, 1024)
                    )
                    top2_edge_probes['logits_1024'] = self._compute_semantic_edge_probe(logits_classes_ch_1024)

                # Site 1: 1024 resolution vs NAMLab prior
                w_sub = top2_cfg.get('logits_1024_namlab', 0.0)
                if w_sub > 0 and output_hub.get('logits_1024') is not None and nam_edge is not None:
                    loss_sub = self.top2_edge_loss(top2_edge_probes['logits_1024'], nam_edge, valid_mask)
                    loss_dict['top2_nam_1024'] = s_align * loss_sub.item()
                    loss_dict['total'] += s_align * w_sub * loss_sub

                # Site 2: 1024 resolution vs Depth prior
                w_sub = top2_cfg.get('logits_1024_depth', 0.0)
                if w_sub > 0 and output_hub.get('logits_1024') is not None and dep_edge is not None:
                    loss_sub = self.top2_edge_loss(top2_edge_probes['logits_1024'], dep_edge, valid_mask)
                    loss_dict['top2_dep_1024'] = s_align * loss_sub.item()
                    loss_dict['total'] += s_align * w_sub * loss_sub
            
            # Aggregation at 256 level
            if output_hub.get('logits_256') is not None:
                if 'logits_256' not in top2_edge_probes and output_hub.get('logits_256') is not None:
                    logits_classes_ch_256 = self._assemble_semantic_logits(
                        output_hub['logits_256'], chunk_voc_ids, resolution=(256, 256)
                    )
                    top2_edge_probes['logits_256'] = self._compute_semantic_edge_probe(logits_classes_ch_256)

                # Site 3: 256 resolution vs NAMLab prior
                w_sub = top2_cfg.get('logits_256_namlab', 0.0)
                if w_sub > 0 and output_hub.get('logits_256') is not None and nam_edge is not None:
                    p_edge_input = F.interpolate(nam_edge, size=(256, 256), mode='bilinear', align_corners=False)
                    v_mask_input = F.interpolate(valid_mask, size=(256, 256), mode='nearest') if valid_mask is not None else None
                    loss_sub = self.top2_edge_loss(top2_edge_probes['logits_256'], p_edge_input, v_mask_input)
                    loss_dict['top2_nam_256'] = s_align * loss_sub.item()
                    loss_dict['total'] += s_align * w_sub * loss_sub

                # Site 4: 256 resolution vs Depth prior
                w_sub = top2_cfg.get('logits_256_depth', 0.0)
                if w_sub > 0 and output_hub.get('logits_256') is not None and dep_edge is not None:
                    p_edge_input = F.interpolate(dep_edge, size=(256, 256), mode='bilinear', align_corners=False)
                    v_mask_input = F.interpolate(valid_mask, size=(256, 256), mode='nearest') if valid_mask is not None else None
                    loss_sub = self.top2_edge_loss(top2_edge_probes['logits_256'], p_edge_input, v_mask_input)
                    loss_dict['top2_dep_256'] = s_align * loss_sub.item()
                    loss_dict['total'] += s_align * w_sub * loss_sub

        # --- [New Loss 2] Boundary-Thinning self-constraint ---
        if self.boundary_thinning_loss and chunk_voc_ids is not None:
            thin_cfg = self.weights['boundary_thinning']
            tau = thin_cfg['threshold']

            assert output_hub, "[Error] In losses.py Co2SAMCriterion forward: output_hub cannot be None!"

            # 1. Process 1024 resolution
            w_1024 = thin_cfg.get('logits_1024', 0.0)
            if w_1024 > 0 and output_hub.get('logits_1024') is not None:
                if 'logits_1024' not in top2_edge_probes:
                    logits_classes_ch_1024 = self._assemble_semantic_logits(
                        output_hub['logits_1024'], chunk_voc_ids, resolution=(1024, 1024)
                    )
                    top2_edge_probes['logits_1024'] = self._compute_semantic_edge_probe(logits_classes_ch_1024)
                    
                        
                l_sub = self.boundary_thinning_loss(top2_edge_probes['logits_1024'], tau, valid_mask)
                loss_dict['thin_1024'], loss_dict['total'] = s_thin * l_sub.item(), loss_dict['total'] + s_thin * w_1024 * l_sub

            # 2. Process 256 resolution
            w_256 = thin_cfg.get('logits_256', 0.0)
            if w_256 > 0 and output_hub.get('logits_256') is not None:
                if 'logits_256' not in top2_edge_probes:
                    logits_classes_ch_256 = self._assemble_semantic_logits(
                        output_hub['logits_256'], chunk_voc_ids, resolution=(256, 256)
                    )
                    top2_edge_probes['logits_256'] = self._compute_semantic_edge_probe(logits_classes_ch_256)
                v_mask_256 = F.interpolate(valid_mask, size=(256, 256), mode='nearest') if valid_mask is not None else None
                l_sub = self.boundary_thinning_loss(top2_edge_probes['logits_256'], tau, v_mask_256)
                loss_dict['thin_256'], loss_dict['total'] = s_thin * l_sub.item(), loss_dict['total'] + s_thin * w_256 * l_sub

        return loss_dict

    def _assemble_semantic_logits(self, logits_list, voc_class_ids, resolution=(1024, 1024)):
        """
        [Internal operator] Semantic projection aggregation: map discrete instance logits
        onto a num_classes-channel semantic canvas.

        This function simulates the output structure of multi-class semantic segmentation.
        It projects the N instance masks in the current chunk onto num_classes channels
        via Scatter Max logic, according to the VOC category IDs provided by DINO.
        Simultaneously, this function constructs Index 0 (background channel) to ensure
        that Top2 competition can trigger at "object-background" boundaries.

        Args:
            logits_list (List[torch.Tensor]): List of instance logits, each element of shape [1, H, W].
            voc_class_ids (torch.Tensor): Corresponding VOC category IDs (1 ~ num_classes-1), shape [N].
            resolution (tuple): Canvas resolution.

        Returns:
            torch.Tensor: Aggregated semantic logits of shape [1, num_classes, H, W], ready for Softmax.
        """
        device = logits_list[0].device
        H, W = resolution
        # 1. Construct foreground channel list (1 ~ num_classes-1)
        fg_list = []
        for c in range(1, self.channels):
            # Find all instance indices for current category c
            matched_idx = (voc_class_ids == c).nonzero(as_tuple=True)[0]
            if len(matched_idx) > 0:
                # Aggregate: stack then take max, keeping the computation graph version stable throughout
                c_logits = torch.stack([logits_list[j].squeeze(0) for j in matched_idx])
                fg_list.append(c_logits.max(dim=0).values)
            else:
                # Fill with extreme negative values to ensure probability → 0 after Softmax
                fg_list.append(torch.full((H, W), fill_value=-100.0, device=device))
        
        # 2. Stack foreground and construct background channel (Index 0)
        fg_stack = torch.stack(fg_list, dim=0) # [num_classes-1, H, W]
        fg_max, _ = fg_stack.max(dim=0)
        # Anchor background to 0.0 baseline level
        bg_channel = torch.zeros((1, H, W), device=device)

        # 3. Final concatenation and return [1, num_classes, H, W]
        return torch.cat([bg_channel, fg_stack], dim=0).unsqueeze(0)

    def _compute_semantic_edge_probe(self, logits):
        """
        [Internal operator] Extract the Top1-Top2 competition probe from semantic logits.

        This operator captures the "uncertainty zone" of the model at category boundaries.
        The smaller the margin, the more intense the competition and the higher the likelihood
        of a boundary. This map serves as the shared input for both BoundaryThinningLoss
        and Top2EdgeAlignmentLoss.

        Args:
            logits (torch.Tensor): num_classes-channel semantic logits, shape [1, num_classes, H, W].

        Returns:
            torch.Tensor: Normalized Semantic Edge Probe [1, 1, H, W], response values in [0, 1].
        """
        # 1. Semantic Edge Extraction
        # Use Softmax to map num_classes channels into probability space, forcing inter-class competition
        probs = F.softmax(logits, dim=1)

        # Neighborhood smoothing: stabilize the probability distribution via 3x3 average pooling
        # to prevent decision noise from fragmenting edges. This step makes the extracted
        # semantic edge more spatially continuous, better matching the Gaussian-blurred priors
        probs = F.avg_pool2d(probs, kernel_size=3, stride=1, padding=1)
        
        # Extract the top two categories by probability
        # values shape: [N, 2, H, W]
        values, _ = torch.topk(probs, k=2, dim=1)
        
        p1 = values[:, 0, :, :] # Top-1 probability
        p2 = values[:, 1, :, :] # Top-2 probability
        
        # Compute confidence margin: smaller margin → more intense competition → stronger edge
        # semantic_edge shape: [N, 1, H, W]
        semantic_edge = (1.0 - (p1 - p2)).unsqueeze(1)
        
        # 4. Map to edge response (1.0 represents the most intense competition, i.e., theoretical boundary)
        return semantic_edge
