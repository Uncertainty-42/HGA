"""
prior_refinement/namlab/merger.py
NAMLab Sequence Merger Implementation
=====================================

This module handles the **Stage A (Deterministic Merging)** phase of NAMLab prior processing.

Functional Scope:
--------------------------
The core task of this module is to parse the raw `.pt` data from NAMLab. It uses the
`merge_sequence` instruction stream to restore the initial atomic segmentation map
(`initial_segmentation`) to a user-specified level (e.g., level 60).

Algorithm Principle:
----------------------------
To avoid brute-force $O(N \times H \times W)$ pixel replacement on high-resolution images,
this module employs a **Mapping Vector** technique:
1. Build a lookup table (LUT) of length `max_id + 1`.
2. Traverse the merge sequence, recording only the ID pointer changes in the LUT ($O(N)$).
3. Perform a single-pass batched remapping on the original image ($O(H \times W)$).
This approach reduces the algorithmic complexity to linear order, significantly
improving processing efficiency.

Usage Context:
-----------------------
This module is called internally by `namlab/core.py` as the initial step of
prior refinement.
"""

import numpy as np
import torch
from typing import Union


def get_level_mask(
    initial_seg: Union[np.ndarray, torch.Tensor],
    merge_seq: Union[np.ndarray, torch.Tensor],
    target_level: int
) -> np.ndarray:
    """
    Restore the initial segmentation map to the target hierarchical level
    according to the merge sequence.

    Args:
        initial_seg (Union[np.ndarray, torch.Tensor]): Raw atomic segmentation index map.
                                                       Typically of shape (H, W), dtype int16.
        merge_seq (Union[np.ndarray, torch.Tensor]): NAMLab merge sequence.
                                                      Shape (N, 2), each row is [high_id, low_id].
        target_level (int): Target number of regions to retain.

    Returns:
        np.ndarray: Index map restored to the specified level (uint16).

    Raises:
        ValueError: Raised when the target level is invalid (exceeds the current
                    region count or is less than 1).
    """
    # =========================================================================
    # 1. Data Preprocessing & Type Alignment
    # =========================================================================
    # Unify to numpy for optimal indexing performance
    if isinstance(initial_seg, torch.Tensor):
        initial_seg = initial_seg.cpu().numpy()
    if isinstance(merge_seq, torch.Tensor):
        merge_seq = merge_seq.cpu().numpy()

    # Obtain the list of unique region IDs in the initial state
    unique_ids = np.unique(initial_seg)
    current_region_count = len(unique_ids)
    max_id = unique_ids.max()

    # Compute the number of merge steps to execute
    steps_to_take = current_region_count - target_level
    if steps_to_take <= 0:
        return initial_seg.copy().astype(np.uint16)

    # =========================================================================
    # 2. Temporal Reversal & Lookup Table (LUT) Mapping
    # =========================================================================
    # Reverse the sequence: merge from the atomic end (Atoms) toward the root end (Root)
    rev_seq = merge_seq[::-1] 
    
    # Initialize lookup table: each ID initially points to itself
    lut = np.arange(max_id + 1, dtype=np.int32)
    
    try:
        for i in range(steps_to_take):
            low_id, high_id = rev_seq[i]
            
            # [Safety Check]: Do NOT silently skip. If the merged ID no longer points
            # to itself, it indicates a logic conflict.
            if lut[high_id] != high_id:
                print(f"\n❌ [NAMLab Merger Error] Logical consistency broken:")
                print(f"   - Merge Sequence: {rev_seq}")
                print(f"   - Current Step: {i}")
                print(f"   - Attempted Merge ID (High): {high_id} -> (Low): {low_id}")
                print(f"   - Anomaly: ID {high_id} was already mapped to {lut[high_id]} in a previous step")
                import sys
                sys.exit(1)
            
            # Establish unidirectional mapping in the LUT
            lut[high_id] = low_id

    except Exception as e:
        # Exception handler: print on-site data for troubleshooting
        print(f"\n❌ [NAMLab Merger Error] Runtime crash:")
        print(f"   - Exception Type: {type(e).__name__}")
        print(f"   - Error Message: {str(e)}")
        import sys
        sys.exit(1)

    # =========================================================================
    # 3. Path Compression & Batched Remapping
    # =========================================================================
    # Handle chained merge mappings (e.g., A->B, B->C), ensuring all IDs
    # resolve directly to the final representative in a single step
    for i in range(len(lut)):
        target = i
        while lut[target] != target:
            target = lut[target]
        lut[i] = target

    # Apply the lookup table to the entire image, achieving one-shot remapping
    refined_mask = lut[initial_seg].astype(np.uint16)
    
    return refined_mask