# ==============================================================================
# Copyright (c) 2026 HGA Authors. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
# SPDX-License-Identifier: MIT
# Project: HGA (Hierarchical-Geometric Alignment)
# GitHub: https://github.com/Uncertainty-42/HGA
# ==============================================================================

"""
Prior Geometrization and Online Preprocessing Engine.

This module is responsible for bridging raw structural priors and the online alignment 
loss space. It systematically transforms discrete or semi-continuous prior signals into 
fully continuous Boundary Attractor Potential Fields capable of providing dense, 
long-range gradient guidance during network back-propagation.

Core Algorithmic Mechanisms:
----------------------------
1. Hill-Equation Reshaping: Applies a non-linear S-shaped mapping to the extracted 
   depth spatial gradients, exponentially amplifying weak geometric discontinuities while 
   suppressing dominant responses from extreme depth gaps.
2. Scale-Compensated Gaussian Diffusion: Diffuses sharp boundary skeletons into continuous 
   attractor fields using separable Gaussian convolution. A mathematically derived sigma^4 
   compensation factor is applied to counteract standard energy attenuation, ensuring 
   robust gradient slopes far from the true boundaries.

HGA Pipeline Context:
---------------------
- Upstream: The 'namlab_raw' input is ideally a high-quality 2D region index map pre-purified 
  by the 'namlab_refinement' sub-package to eliminate redundant micro-fragment noise.
- Downstream: The synthesized continuous fields ('namlab_blurred', 'depth_blurred') are 
  designed exclusively to be consumed by the 'HGACriterion' ('losses.py') to enforce 
  decision-competition alignment during the forward pass.

Minimal Integration Example (Inside a Custom DataLoader):
---------------------------------------------------------
    >>> from priors_optimization.preprocess import Preprocessor
    >>> 
    >>> # Typical usage inside the '__getitem__' method:
    >>> # namlab_raw and depth_raw should be exactly spatially aligned with the main image
    >>> nam_blurred, dep_blurred = Preprocessor.prepare_processed_priors(
    ...     namlab_raw, 
    ...     depth_raw, 
    ...     sigma=3.5, 
    ...     hill_params=(0.2, 0.7, 1.0)
    ... )
    >>> 
    >>> # Return the processed fields alongside your image and ground truth
    >>> return image, label, nam_blurred, dep_blurred, valid_mask
"""

import torch
import torch.nn.functional as F

class Preprocessor:
    @classmethod
    def prepare_processed_priors(cls, namlab_raw=None, depth_raw=None, sigma=3.5, hill_params=(0.2, 0.7, 2.0)):
        """
        Process raw spatial priors into continuous boundary attractor potential fields.

        This method serves as the central orchestration pipeline for prior processing.
        It performs two distinct branches of operations:
        1. NAMLab Branch: Converts a discrete, categorical region partition mask (index map) 
           into a 0/1 binary boundary skeleton, which is subsequently diffused into a 
           continuous region boundary attractor field via scale-compensated Gaussian blur.
        2. Depth Branch: Standardizes a raw continuous relative depth map, computes its spatial 
           gradients using Sobel operators, non-linearly reshapes the gradient magnitudes using 
           the Hill equation to balance weak and strong boundary cues, and diffuses the reshaped 
           gradients into a continuous depth geometric attractor field.

        Args:
            namlab_raw (Optional[Union[numpy.ndarray, torch.Tensor]]): Raw 2D region index map of shape [H, W], 
                where each pixel contains a categorical region ID (integer-like).
            depth_raw (Optional[Union[numpy.ndarray, torch.Tensor]]): Raw 2D continuous relative depth map 
                of shape [H, W].
            sigma (float): Standard deviation for Gaussian kernel used to diffuse boundaries. 
                Determines the spatial attraction range of the potential field. Default: 3.5.
            hill_params (tuple): Three-element configuration (x_center, y_center, slope_k) 
                controlling the S-shaped mapping of the Hill equation. Default: (0.2, 0.7, 2.0).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - namlab_blurred (torch.Tensor): 2D continuous region boundary attractor field of 
                  shape [H, W] and dtype float32, with values scale-compensated by sigma^4.
                - depth_blurred (torch.Tensor): 2D continuous depth boundary attractor field of 
                  shape [H, W] and dtype float32, with values scale-compensated by sigma^4.
        """
        namlab_blurred, depth_blurred = None, None

        namlab_raw = torch.from_numpy(namlab_raw).float().unsqueeze(0).unsqueeze(0)
        depth_raw = torch.from_numpy(depth_raw).float().unsqueeze(0).unsqueeze(0)
        # ---------------------------------------------------------
        # 1. Process NAMLab: discrete region map -> continuous boundary potential field
        # ---------------------------------------------------------
        if namlab_raw is not None:
            # Ensure channel dimension [N, 1, H, W]
            if namlab_raw.ndim == 3:
                namlab_raw = namlab_raw.unsqueeze(1)
            
            # Step 1: Extract binary discrete boundary (0/1) (computed only once)
            namlab_edges = cls._extract_discrete_boundaries(namlab_raw)

            # Step 2: Generate blurred potential field
            namlab_blurred = cls._apply_gaussian_blur(namlab_edges, sigma).squeeze()
            
        # ---------------------------------------------------------
        # 2. Process Depth: continuous depth map -> gradient boundary potential field
        # ---------------------------------------------------------
        if depth_raw is not None:
            d_tensor = depth_raw.clone() # Clone to avoid in-place parameter pollution

            # Step A: Normalize per image
            d_min, d_max = d_tensor.min(), d_tensor.max()
            if d_max - d_min > 1e-6:
                d_tensor = (d_tensor - d_min) / (d_max - d_min)
            else:
                d_tensor = d_tensor - d_min
            
            # Step B: Compute gradient operator
            d_tensor = cls._compute_sobel_gradient(d_tensor)
            
            # Step C: Hill-equation distribution shift ---
            # Parameter tuple format: (x_center, y_center, slope_k)
            if hill_params is not None:
                xc, yc, k = hill_params
                eps = 1e-8
                # Compute offset coefficient M to make the curve exactly pass through (xc, yc)
                M = (xc / (1.0 - xc + eps)) * (((1.0 - yc) / (yc + eps))**(1.0 / k))
                
                # Hill equation implementation: y = x^k / (x^k + [M(1-x)]^k)
                x_k = d_tensor.pow(k)
                denom_part = (M * (1.0 - d_tensor + eps)).pow(k)
                d_tensor = x_k / (x_k + denom_part + eps)
            
            # Step E: Gaussian blurring
            depth_blurred = cls._apply_gaussian_blur(d_tensor, sigma).squeeze()

        return namlab_blurred, depth_blurred
                
    # ------------------ [Internal Mathematical Operator Engine] ------------------
    @classmethod
    def _extract_discrete_boundaries(cls, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract a 0/1 binary discrete boundary skeleton from a categorical region index map.

        This operator leverages a highly efficient, parameter-free local max-min pooling window.
        For each pixel in the region map, it evaluates the local maximum and minimum values in a 
        3x3 neighborhood. If the local maximum does not equal the local minimum, the pixel lies on 
        a discontinuity (i.e., a boundary between different regions) and is marked as 1.0; 
        otherwise, it is marked as 0.0.

        Args:
            mask (torch.Tensor): 4D categorical region index map tensor of shape [1, 1, H, W] 
                and float32 dtype.

        Returns:
            torch.Tensor: 4D binary boundary skeleton of shape [1, 1, H, W] and float32 dtype, 
                where 1.0 indicates a region boundary and 0.0 indicates a flat internal region.
        """
        x = mask.float()
        # Maximum within 3x3 window
        max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        # Minimum within 3x3 window (implemented via -max(-x))
        min_pool = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        # Where max != min, it is a boundary
        boundary = (max_pool != min_pool).float()
        return boundary

    @classmethod
    def _compute_sobel_gradient(cls, x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized continuous spatial gradients from a continuous input tensor.

        This operator uses standard 3x3 horizontal (Sobel-X) and vertical (Sobel-Y) 2D convolution 
        kernels to compute the partial spatial derivatives G_x and G_y. The gradient magnitude is 
        then evaluated as sqrt(G_x^2 + G_y^2 + eps), where eps prevents division-by-zero errors. 
        Finally, the raw magnitude is min-max normalized internally within the image to strictly 
        constrain the gradient energy to the [0, 1] range, preventing numerical imbalance in the 
        subsequent losses.

        Args:
            x (torch.Tensor): 4D normalized continuous input tensor (e.g., relative depth map) 
                of shape [1, 1, H, W] and float32 dtype.

        Returns:
            torch.Tensor: 4D normalized continuous gradient magnitude of shape [1, 1, H, W] 
                and float32 dtype, with values strictly bounded in [0, 1].
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Strictly compress gradient energy to [0, 1] to prevent loss imbalance
        g_min, g_max = grad_mag.min(), grad_mag.max()
        if g_max - g_min > 1e-6:
            grad_mag = (grad_mag - g_min) / (g_max - g_min)
        else:
            grad_mag = grad_mag - g_min
            
        return grad_mag
    @classmethod
    def _apply_gaussian_blur(cls, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Diffuse discrete or sharp boundaries into a continuous attractor field using 
        separable Gaussian convolution with scale-enhancement compensation.

        To eliminate boundary truncation errors and preserve mathematical continuity, the 
        kernel size is determined dynamically as an odd integer near 8 * sigma. The convolution 
        is executed as a separable horizontal 1D kernel [1, 1, 1, K] followed by a vertical 
        1D kernel [1, 1, K, 1], reducing the computation complexity from O(K^2 * H * W) to 
        O(K * H * W).

        Importantly, a scale-enhancement compensation factor of sigma^4 is multiplied to the output. 
        Because standard Gaussian blur is energy-conserving, its peak energy attenuates at a rate of 
        1/sigma^2 as the blur radius increases. The sigma^4 factor overrides this constraint: 
        the first sigma^2 offsets the energy attenuation, while the second active sigma^2 enhances 
        the central peak and gradients of the attractor potential field as sigma grows, ensuring 
        strong long-range guidance.

        Args:
            x (torch.Tensor): 4D boundary tensor to be diffused, shape [1, 1, H, W] and float32 dtype.
            sigma (float): Standard deviation of the Gaussian distribution. If sigma <= 0, the input 
                is returned unchanged.

        Returns:
            torch.Tensor: 4D scale-compensated continuous attractor field of shape [1, 1, H, W] 
                and float32 dtype.
        """
        if sigma <= 0:
            return x
            
        # Set kernel size heuristically to 8*sigma and ensure it is odd
        k_size = int(8 * sigma + 0.5)
        k_size = k_size + 1 if k_size % 2 == 0 else k_size
        k_size = max(3, k_size)
            
        # Dynamically generate 1D Gaussian kernel
        coords = torch.arange(k_size, dtype=torch.float32, device=x.device)
        coords -= (k_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g_1d /= g_1d.sum()

        g_h = g_1d.view(1, 1, 1, k_size)
        g_v = g_1d.view(1, 1, k_size, 1)
    
        # 3. Perform separable convolution
        blurred = F.conv2d(x, g_h, padding=(0, k_size // 2))
        blurred = F.conv2d(blurred, g_v, padding=(k_size // 2, 0))
        
        # 4. Apply scale compensation factor
        # Multiply by sigma^4 so that larger sigma enhances both potential field intensity and gradient magnitude globally
        # And points farther from the origin receive stronger enhancement
        return blurred * (sigma ** 4)