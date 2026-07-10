# ==============================================================================
# Copyright (c) 2026 HGA Authors. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
# SPDX-License-Identifier: MIT
# Project: HGA (Hierarchical-Geometric Alignment)
# GitHub: https://github.com/Uncertainty-42/HGA
# ==============================================================================

"""
Hierarchical-Geometric Alignment (HGA) Core Loss Module.

This module provides a standalone, plug-and-play toolkit containing the core loss
formulations of the HGA paradigm. It contains individual boundary alignment and
thinning loss operators, managed by a unified facade criterion ('HGACriterion'):

1.  Top2EdgeAlignmentLoss: Attractor-field-based decision boundary alignment.
2.  BoundaryThinningLoss: Local energy budget self-constraint to sparsify boundaries.
3.  HGACriterion: The main plug-and-play facade/arbiter for easy host model integration.

Typical Integration Flow:
--------------------------
1. Preprocess your raw structural priors (NAMLab region masks and Depth maps)
   using the 'Preprocessor' module to generate continuous attractor fields.
2. Instantiate 'HGACriterion' in your training script, passing your dataset's
   class count and specific HGA loss weights.
3. During the forward pass, feed the upsampled multi-class logits and the 
   preprocessed attractor fields into the criterion.
4. Accumulate the returned loss values into your base training objective and backpropagate.

Minimal Integration Example:
----------------------------
    >>> from priors_optimization.losses import HGACriterion
    >>>
    >>> # 1. Initialize the joint criterion
    >>> criterion = HGACriterion(
    ...     use_priors=True,
    ...     w_align_nam=0.2,
    ...     w_align_dep=0.2,
    ...     w_thin=0.2,
    ...     threshold=10.0,
    ...     num_classes=21  # 21 for Pascal VOC, 81 for MS COCO
    ... )
    >>>
    >>> # 2. Inside the training loop
    >>> # segs: [B, C, H, W] multi-class logits (interpolated to match prior size)
    >>> # nam_blurred, dep_blurred, valid_mask: [B, H, W] or [B, 1, H, W]
    >>> hga_losses = criterion(segs, nam_blurred, dep_blurred, valid_mask)
    >>>
    >>> # 3. Accumulate and backpropagate
    >>> loss = base_loss + sum(hga_losses.values())
    >>> loss.backward()
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import nn


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
    

class HGACriterion(nn.Module):
    """
    HGACriterion: Unified loss computation arbiter for Hierarchical-Geometric Alignment (HGA).

    This class acts as an arbiter/facade that encapsulates individual HGA loss components,
    specifically the Prior-Guided Decision-Competition Alignment Loss (for NAMLab and Depth priors)
    and the Boundary-Thinning self-constraint Loss. The host model's training loop can interface
    with this module directly, receiving a dictionary of weighted loss items.

    Key Design Principles:
    1.  **Decoupled & Plug-and-Play**: It operates directly on the output multi-class logits,
        requiring no modifications to the host backbone, decoder, or training pipelines.
    2.  **Generic Category Support**: Dynamically accepts a configurable class count (`num_classes`),
        allowing seamless transitions between datasets (e.g., Pascal VOC with 21 classes,
        MS COCO with 81 classes, or Cityscapes with 19 classes).
    3.  **Zero-Overhead Early Exit**: If prior-guided training is disabled (`use_priors=False`) or
        all loss weights are set to 0.0, the module bypasses calculations in the forward pass,
        returning zero losses with no additional VRAM or compute overhead.
    """

    def __init__(
        self,
        use_priors: bool = False,
        w_align_nam: float = 0.0,
        w_align_dep: float = 0.0,
        w_thin: float = 0.0,
        threshold: float = 10.0,
        num_classes: int = 21
    ):
        """
        Initialize HGACriterion with robust parameter verification and conditional instantiation.

        Args:
            use_priors (bool): Global switch to enable or disable HGA prior-guided constraints.
            w_align_nam (float): Weight for the NAMLab prior-guided decision-competition alignment loss.
            w_align_dep (float): Weight for the depth prior-guided decision-competition alignment loss.
            w_thin (float): Weight for the boundary-thinning self-constraint loss.
            threshold (float): Local uncertainty energy budget threshold (tau) for boundary thinning.
            num_classes (int): Total number of semantic categories (including background) for the dataset.

        Raises:
            TypeError: If any weight/threshold is not numeric, or if use_priors/num_classes are invalid types.
            ValueError: If any loss weight is negative, the threshold is non-positive,
                        or num_classes is less than 2.
        """
        super().__init__()

        # --- Strict Parameter Verification ---
        if not isinstance(use_priors, bool):
            raise TypeError(f"use_priors must be a boolean, but received: {type(use_priors)}")

        if not isinstance(num_classes, int):
            raise TypeError(f"num_classes must be an integer, but received: {type(num_classes)}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, but received: {num_classes}")

        weights = {
            "w_align_nam": w_align_nam,
            "w_align_dep": w_align_dep,
            "w_thin": w_thin
        }
        for name, val in weights.items():
            if not isinstance(val, (int, float)):
                raise TypeError(f"Loss weight '{name}' must be numeric, but received: {type(val)}")
            if val < 0.0:
                raise ValueError(f"Loss weight '{name}' must be non-negative, but received: {val}")

        if not isinstance(threshold, (int, float)):
            raise TypeError(f"Thinning threshold must be numeric, but received: {type(threshold)}")
        if threshold <= 0.0:
            raise ValueError(f"Thinning threshold must be strictly positive, but received: {threshold}")

        # --- Attribute Assignment ---
        self.use_priors = use_priors
        self.w_align_nam = float(w_align_nam)
        self.w_align_dep = float(w_align_dep)
        self.w_thin = float(w_thin)
        self.threshold = float(threshold)
        self.channels = num_classes

        # --- Conditional Low-level Operator Instantiation ---
        # Instantiate operators only when active to maintain zero VRAM/compute overhead when disabled
        self.top2_edge_loss = (
            Top2EdgeAlignmentLoss()
            if (self.use_priors and (self.w_align_nam > 0.0 or self.w_align_dep > 0.0))
            else None
        )
        self.boundary_thinning_loss = (
            BoundaryThinningLoss()
            if (self.use_priors and self.w_thin > 0.0)
            else None
        )
    
    def forward(
        self,
        segs: torch.Tensor,
        namlab_blurred: Optional[torch.Tensor] = None,
        depth_blurred: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Execute joint computation of prior-guided alignment and boundary-thinning losses.

        This method extracts a Semantic Edge Probe from the multi-class segmentation logits,
        and subsequently enforces decision-competition alignment constraints using the provided
        blurred NAMLab and depth prior attractor fields.

        Expected Tensor Shapes:
            - segs: [B, num_classes, H, W] raw multi-class logits.
            - namlab_blurred: [B, H, W] or [B, 1, H, W] continuous NAMLab attractor potential field.
            - depth_blurred: [B, H, W] or [B, 1, H, W] continuous depth gradient attractor potential field.
            - valid_mask: [B, H, W] or [B, 1, H, W] binary spatial geometric mask (1 for valid, 0 for padding).

        Defensive Alignment & Safety Mechanics:
            1.  **Auto-Unsqueezing (Dimensional Alignment)**: In standard dataloading pipelines,
                prior potential fields and validity masks are frequently stacked as 3D tensors [B, H, W].
                This method automatically detects 3D structures and non-destructively unsqueezes them to
                4D [B, 1, H, W] tensors to satisfy spatial mapping operator requirements and prevent
                multi-GPU/DDP dimension mismatches.
            2.  **Null-Safety Gateways**: If a specific prior is disabled or fails to load (i.e., passed
                as `None`), its associated calculations are safely bypassed, keeping its respective
                loss term at `0.0` without triggering runtime exceptions.
            3.  **Corrected Early-Exit Fuse**: If global prior training is disabled (`use_priors=False`) or
                all individual loss weights are configured to 0.0, the function triggers a rapid early exit.
                It returns a structurally complete dictionary of zero-valued tensors allocated on the same
                device/dtype as the inputs, avoiding any graph tracking or VRAM allocation overhead.

        Args:
            segs (torch.Tensor): Multi-class segmentation logits, shape [B, num_classes, H, W].
            namlab_blurred (torch.Tensor, optional): Blurred NAMLab prior, shape [B, H, W] or [B, 1, H, W].
            depth_blurred (torch.Tensor, optional): Blurred depth prior, shape [B, H, W] or [B, 1, H, W].
            valid_mask (torch.Tensor, optional): Spatial validity mask, shape [B, H, W] or [B, 1, H, W].

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing weighted individual loss components:
                - "loss_align_nam": Weighted NAMLab alignment loss (scalar tensor).
                - "loss_align_dep": Weighted depth alignment loss (scalar tensor).
                - "loss_thin": Weighted boundary-thinning constraint (scalar tensor).
        """
        # Initialize default zero-valued loss tensors mapped to the input device and dtype
        loss_dict = {
            "loss_align_nam": torch.tensor(0.0, device=segs.device, dtype=segs.dtype),
            "loss_align_dep": torch.tensor(0.0, device=segs.device, dtype=segs.dtype),
            "loss_thin": torch.tensor(0.0, device=segs.device, dtype=segs.dtype)
        }

        # 1. Zero-Overhead Early Exit (Corrected to inspect all active weights)
        if not self.use_priors or (self.w_align_nam <= 0.0 and self.w_align_dep <= 0.0 and self.w_thin <= 0.0):
            return loss_dict

        # 2. Defensive Dimensional Alignment (Unsqueeze 3D tensors to 4D to ensure broadcasting compatibility)
        if valid_mask is not None and valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)

        if namlab_blurred is not None and namlab_blurred.ndim == 3:
            namlab_blurred = namlab_blurred.unsqueeze(1)

        if depth_blurred is not None and depth_blurred.ndim == 3:
            depth_blurred = depth_blurred.unsqueeze(1)

        # 3. Extract Shared Semantic Edge Probe (Shape: [B, 1, H, W])
        semantic_edge = self._compute_semantic_edge_probe(segs)

        # 4. Compute Weighted Decision-Competition Alignment Losses (With Null-Safety Gates)
        if self.top2_edge_loss:
            if self.w_align_nam > 0.0 and namlab_blurred is not None:
                loss_align_nam = self.top2_edge_loss(semantic_edge, namlab_blurred, valid_mask)
                loss_dict["loss_align_nam"] = self.w_align_nam * loss_align_nam

            if self.w_align_dep > 0.0 and depth_blurred is not None:
                loss_align_dep = self.top2_edge_loss(semantic_edge, depth_blurred, valid_mask)
                loss_dict["loss_align_dep"] = self.w_align_dep * loss_align_dep

        # 5. Compute Weighted Boundary-Thinning Self-Constraint Loss
        if self.boundary_thinning_loss and self.w_thin > 0.0:
            loss_thin = self.boundary_thinning_loss(semantic_edge, self.threshold, valid_mask)
            loss_dict["loss_thin"] = self.w_thin * loss_thin

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
