"""
prior_refinement/namlab/utils.py
NAMLab Structural Utilities
======================================

This module handles the **Stage B (Feature Extraction & Structure Construction)**
phase of NAMLab prior processing.

Functional Scope:
--------------------------
This module provides core tools for extracting attributes from image data.
It aggregates pixel-level color and depth information into region-level statistics,
and constructs a structural adjacency graph describing spatial relationships
between regions.

Core Technologies:
---------------------------
1. **Vectorized Aggregation**: Uses `np.bincount` in place of Python loops,
   achieving extremely fast per-region mean computation.
2. **4-Connectivity Shifting**: Identifies all neighboring region pairs across the
   entire image in $O(H \cdot W)$ complexity via image shifting comparison, and
   precisely counts shared boundary pixel counts.
3. **D50 Color Standardization**: Strictly conforms to the NAMLab paper specification,
   supporting Lab color space conversion under the D50 illuminant.

Usage Context:
-----------------------
This module is called by `namlab/core.py` and provides the necessary decision
factors for the subsequent Stage C (Fragment Purification).
"""

import numpy as np
from typing import Dict, Tuple


def convert_rgb_to_lab(
    img_rgb: np.ndarray, 
    illuminant: str = 'D50', 
    lib: str = 'skimage'
) -> np.ndarray:
    """
    Convert an RGB image to the Lab color space.

    Args:
        img_rgb (np.ndarray): Raw RGB image, range [0, 255], shape (H, W, 3).
        illuminant (str): Standard illuminant type. NAMLab defaults to 'D50'.
        lib (str): Third-party library to use. Currently supports 'skimage'.

    Returns:
        np.ndarray: Lab color space image, shape (H, W, 3).
    """
    if lib == 'skimage':
        from skimage import color
        # skimage.color.rgb2lab accepts input in range [0, 1] or [0, 255]
        return color.rgb2lab(img_rgb, illuminant=illuminant)
    else:
        raise NotImplementedError(f"[NAMLab Utils] Library '{lib}' is not supported yet.")


def extract_region_features(
    mask: np.ndarray, 
    img_lab: np.ndarray, 
    depth_norm: np.ndarray
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute the mean color (Lab) and mean depth for each region ID.

    This function leverages numpy.bincount for high-performance weighted statistics.

    Args:
        mask (np.ndarray): Index map, shape (H, W), dtype uint16.
        img_lab (np.ndarray): Lab image, shape (H, W, 3).
        depth_norm (np.ndarray): Normalized depth map, shape (H, W), range [0, 1].

    Returns:
        Dict[int, Dict[str, np.ndarray]]: 
            Key is region ID, Value is a dictionary containing 'color' (3,)
            and 'depth' (1,) mean values.
    """
    # =========================================================================
    # 1. Preparation
    # =========================================================================
    flat_mask = mask.ravel()
    unique_ids = np.unique(flat_mask)
    max_id = unique_ids.max()
    
    # Total pixel count per region
    pixel_counts = np.bincount(flat_mask, minlength=max_id + 1)
    # Filter out non-existent IDs (count=0) to avoid division by zero
    valid_mask = pixel_counts > 0
    
    features = {}

    # =========================================================================
    # 2. Vectorized Mean Calculation
    # =========================================================================
    # Compute the mean of three color channels
    avg_lab = np.zeros((max_id + 1, 3))
    for c in range(3):
        channel_sum = np.bincount(flat_mask, weights=img_lab[..., c].ravel(), minlength=max_id + 1)
        avg_lab[valid_mask, c] = channel_sum[valid_mask] / pixel_counts[valid_mask]

    # Compute the depth channel mean
    avg_depth = np.zeros(max_id + 1)
    depth_sum = np.bincount(flat_mask, weights=depth_norm.ravel(), minlength=max_id + 1)
    avg_depth[valid_mask] = depth_sum[valid_mask] / pixel_counts[valid_mask]

    # =========================================================================
    # 3. Results Packaging
    # =========================================================================
    for rid in unique_ids:
        features[rid] = {
            'color': avg_lab[rid],
            'depth': avg_depth[rid],
            'area': pixel_counts[rid]
        }
        
    return features


def build_adjacency_graph(mask: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Build a region adjacency table using 4-neighborhood shifting and count
    shared boundary lengths.

    Args:
        mask (np.ndarray): Index map, shape (H, W).

    Returns:
        Dict[Tuple[int, int], int]: 
            Key is an ordered ID pair (min_id, max_id), Value is the shared
            boundary pixel count.
    """
    # =========================================================================
    # 1. Extracting Pairs (Vertical and Horizontal Directions)
    # =========================================================================
    # Horizontal adjacency: pixel vs. its right neighbor
    pairs_h_left = mask[:, :-1].ravel()
    pairs_h_right = mask[:, 1:].ravel()
    
    # Vertical adjacency: pixel vs. its bottom neighbor
    pairs_v_top = mask[:-1, :].ravel()
    pairs_v_bottom = mask[1:, :].ravel()
    
    # Combine all potential boundary pairs
    all_left = np.concatenate([pairs_h_left, pairs_v_top])
    all_right = np.concatenate([pairs_h_right, pairs_v_bottom])
    
    # =========================================================================
    # 2. Boundary Filtering
    # =========================================================================
    # A boundary exists only when the IDs of two adjacent pixels differ
    is_boundary = all_left != all_right
    b_left = all_left[is_boundary]
    b_right = all_right[is_boundary]
    
    # Sort each pair so that (ID_A, ID_B) and (ID_B, ID_A) are treated
    # as the same edge
    stacked_pairs = np.stack([np.minimum(b_left, b_right), 
                               np.maximum(b_left, b_right)], axis=1)
    
    # Use np.unique to count occurrences of all unique ID pairs in one shot
    # (i.e., the shared boundary pixel count)
    unique_pairs, counts = np.unique(stacked_pairs, axis=0, return_counts=True)
    
    # =========================================================================
    # 3. Final Mapping
    # =========================================================================
    adjacency = {}
    for pair, count in zip(unique_pairs, counts):
        adjacency[tuple(pair)] = count
        
    return adjacency