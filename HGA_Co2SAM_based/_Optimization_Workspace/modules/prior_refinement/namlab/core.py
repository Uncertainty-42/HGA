"""
prior_refinement/namlab/core.py
NAMLab Refinement Core Orchestrator
===================================

This module is the **Core Orchestrator** of the NAMLab prior refinement system.

Functional Scope:
-------------------------
This module serves as the unified facade interface of the `namlab` subdirectory,
responsible for coordinating and executing the complete lifecycle of prior generation.
It integrates the discrete Stage A (Sequential Merging), Stage B (Attribute Extraction),
and Stage C (Fragment Purification) into a single pipeline.

Core Responsibilities:
------------------------------
1. **Pipeline Execution**: Sequentially dispatches the merger, utils, and purifier modules
   in logical order.
2. **Depth Pre-processing**: Performs Min-Max normalization of the depth map to resolve
   negative value offsets and inter-image scale inconsistencies.
3. **Parameter Routing**: Parses the global `config` object and distributes the corresponding
   hyperparameters to the underlying algorithmic functions.

Call Flow:
-------------------
External (priors_manager) -> core.run_namlab_refinement
    -> Stage A: merger.get_level_mask
    -> Pre-process: utils.convert_rgb_to_lab & Depth Normalization
    -> Stage B: utils.extract_region_features & utils.build_adjacency_graph
    -> Stage C: purifier.refine_small_regions
"""

import numpy as np
from typing import Dict, Any, Union

# Import internal functional components
from . import merger
from . import utils
from . import purifier


def run_namlab_refinement(
    pt_data: Dict[str, Any], 
    img_rgb: np.ndarray, 
    depth_raw: np.ndarray, 
    config: Any
) -> np.ndarray:
    """
    Execute the full-pipeline workflow of NAMLab prior refinement.

    Args:
        pt_data (Dict): Data loaded from the raw .pt file.
                         Expected to contain 'initial_segmentation' and 'merge_sequence'.
        img_rgb (np.ndarray): Original image (H, W, 3), range [0, 255].
        depth_raw (np.ndarray): Raw depth map (H, W), may contain negative values.
        config (Any): Configuration object (Box/Dict), must contain priors.namlab
                      and priors.depth branches.

    Returns:
        np.ndarray: The final index map (uint16) after hierarchical coherence refinement.
    """
    # =========================================================================
    # 1. Configuration Parsing
    # =========================================================================
    nam_cfg = config.priors.namlab
    dep_cfg = config.priors.depth

    # =========================================================================
    # 2. Stage A: Base-Level Restoration (Merge Sequence Restoration)
    # =========================================================================
    # Merge the initial atomic regions according to the NAMLab-preset sequence
    # to the target level (e.g., 60)
    base_mask = merger.get_level_mask(
        initial_seg=pt_data['initial_segmentation'],
        merge_seq=pt_data['merge_sequence'],
        target_level=nam_cfg.target_level
    )

    # Directly return the raw NAMLab region map (e.g., 60) without purification
    if nam_cfg.get("purification_enabled", True) == False:
        return base_mask

    # =========================================================================
    # 3. Data Pre-processing
    # =========================================================================
    # A. Color space conversion (strictly conforming to D50 standard)
    img_lab = utils.convert_rgb_to_lab(
        img_rgb=img_rgb,
        illuminant=nam_cfg.color_logic.illuminant,
        lib=nam_cfg.color_logic.lib
    )

    # B. Depth map normalization (Min-Max scaling)
    # Resolve the negative value offsets and inter-image scale discrepancies
    # inherent in Depth Anything
    d_min = depth_raw.min()
    d_max = depth_raw.max()
    if d_max > d_min:
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        # Prevent division by zero for all-black or all-white depth maps
        depth_norm = np.zeros_like(depth_raw)
    

    # =========================================================================
    # 4. Stage B: Feature & Structure Extraction
    # =========================================================================
    # Extract per-region Lab color means and depth means
    region_features = utils.extract_region_features(
        mask=base_mask,
        img_lab=img_lab,
        depth_norm=depth_norm
    )

    # Build the region adjacency graph and count shared boundary pixel counts
    adjacency_graph = utils.build_adjacency_graph(mask=base_mask)

    # =========================================================================
    # 5. Stage C: Hierarchical Fragment Purification
    # =========================================================================
    # Use the cost formula to merge undersized fragments and recover
    # broken semantic associations
    refined_mask = purifier.refine_small_regions(
        mask=base_mask,
        features=region_features,
        adjacency=adjacency_graph,
        strategy=nam_cfg.area_constraint.strategy,
        threshold=nam_cfg.area_constraint.threshold,
        epsilon=nam_cfg.cost_params.epsilon
    )

    return refined_mask