"""
prior_refinement/namlab/purifier.py
NAMLab Small Region Purifier
============================

This module handles the **Stage C (Fragment Purification & Contour-Based Merging)**
phase of NAMLab prior processing.

Functional Scope:
--------------------------
This module identifies "numerically unstable fragments" in the index map based on
area constraints, and greedily merges them into their most compatible
neighbor regions using color similarity, depth consistency, and geometric contour
relationships.

Core Algorithm:
------------------------
1. **Adaptive Strategy**: Supports both 'pxl' (absolute pixels) and 'ratio'
   (proportion of whole image) area decision logic.
2. **Merging Cost Formula**: Implements the formula
   $\mathcal{C} = \frac{\Delta \text{Color} \times \Delta \text{Depth}}{\text{SharedBoundary} + \epsilon}$.
3. **Dynamic Graph Update**: When a fragment is merged, the system updates the
   mean features of its neighbors and the global contour connection state
   in real time, ensuring subsequent merge decisions are based on the latest
   master.

Usage Context:
-----------------------
This module is called by `namlab/core.py` and constitutes the final step in
generating the refined index map.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_merging_cost(
    feat_i: Dict[str, Any], 
    feat_j: Dict[str, Any], 
    shared_boundary: int, 
    epsilon: float = 1e-6
) -> float:
    """
    Compute the merging cost between two regions.

    The lower the cost, the closer the two regions are.

    Args:
        feat_i (Dict): Features of region i (contains 'color', 'depth').
        feat_j (Dict): Features of region j (contains 'color', 'depth').
        shared_boundary (int): Length of shared boundary pixels between the two regions.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The computed merging cost.
    """
    # 1. Compute Lab color space Euclidean distance (Color Dissimilarity)
    delta_c = np.linalg.norm(feat_i['color'] - feat_j['color'])
    
    # 2. Compute absolute difference in normalized depth (Depth Dissimilarity)
    delta_d = np.abs(feat_i['depth'] - feat_j['depth'])
    
    # 3. Apply the cost formula: (ΔColor * ΔDepth) / (SharedBoundary + ε)
    # Logic: the more similar the color/depth and the tighter the contact,
    # the lower the cost
    cost = (delta_c * delta_d) / (shared_boundary + epsilon)
    
    return float(cost)


def refine_small_regions(
    mask: np.ndarray, 
    features: Dict[int, Dict[str, Any]], 
    adjacency: Dict[Tuple[int, int], int], 
    strategy: str = 'pxl', 
    threshold: float = 4700,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Execute fragment purification logic: merge undersized regions into their
    optimal neighbor.

    Args:
        mask (np.ndarray): Base index map generated in Stage A.
        features (Dict): Region features extracted in Stage B.
        adjacency (Dict): Region adjacency table extracted in Stage B.
        strategy (str): Area decision strategy ('pxl' or 'ratio').
        threshold (float): Area threshold.
        epsilon (float): Cost computation tolerance.

    Returns:
        np.ndarray: Final index map after purification (uint16).
    """
    # =========================================================================
    # 1. Initialize Management Data (Preparation)
    # =========================================================================
    h, w = mask.shape
    total_pixels = h * w
    
    # Determine the actual pixel threshold according to the strategy
    area_limit = threshold if strategy == 'pxl' else threshold * total_pixels
    
    # Build a mapping lookup table, initially each ID points to itself.
    # Uses Union-Find thinking, but managed here via dynamic mapping.
    parent_map = {rid: rid for rid in features.keys()}
    
    # Restructure the adjacency table into a "neighbor dictionary" for fast lookup
    # neighbors[id] = {neighbor_id: shared_boundary_length}
    neighbor_map = {}
    for (id1, id2), length in adjacency.items():
        if id1 not in neighbor_map: neighbor_map[id1] = {}
        if id2 not in neighbor_map: neighbor_map[id2] = {}
        neighbor_map[id1][id2] = length
        neighbor_map[id2][id1] = length

    # =========================================================================
    # 2. Identify and Sort Fragments (Small Region Identification)
    # =========================================================================
    # Sort by area ascending; process the most granular fragments first
    small_regions = [rid for rid, feat in features.items() if feat['area'] < area_limit]
    small_regions.sort(key=lambda rid: features[rid]['area'])

    # =========================================================================
    # 3. Greedy Merging Loop
    # =========================================================================
    for rid in small_regions:
        # [Safety Check 1]: If the current fragment was already merged as a
        # neighbor in a previous loop, skip it
        if rid not in neighbor_map:
            continue

        # # [Probe 1] Monitor the current fragment being processed
        # print(f"👉 Processing RID:{rid:4d} | Area:{features[rid]['area']:5.0f}", end=" ")
            
        # Obtain the active neighbors of the current fragment
        current_neighbors = neighbor_map[rid].copy()
        if not current_neighbors:
            continue  # Isolated region; leave as-is
            
        # Find the neighbor with the minimal cost
        best_neighbor = -1
        min_cost = float('inf')
        
        for nid, shared_len in current_neighbors.items():
            # [Safety Check 2]: Ensure the candidate neighbor still exists
            # in the global graph
            if nid not in neighbor_map:
                continue
            # [Dynamic State Check]: If this region's area has already reached
            # the threshold due to previous merges, retain its ID; do not
            # merge into another neighbor
            if features[rid]['area'] >= area_limit:
                continue
            
            cost = calculate_merging_cost(features[rid], features[nid], shared_len, epsilon)
            if cost < min_cost:
                min_cost = cost
                best_neighbor = nid
        
        # If no active neighbor was found after searching, skip
        if best_neighbor == -1:
            continue

        # # [Probe 2] Monitor the merge target and cost
        # print(f"-> Target:{best_neighbor:4d} | TargetArea:{features[best_neighbor]['area']:5.0f} | Cost:{min_cost:.6f}")

        try:
            # Execute merge logic: merge rid into best_neighbor
            # A. Update Parent mapping
            parent_map[rid] = best_neighbor
            
            # B. Update Neighbor features (weighted average) — holistic state-machine update
            old_neighbor_area = features[best_neighbor]['area']
            rid_area = features[rid]['area']
            new_area = old_neighbor_area + rid_area
            
            # Dynamic evolution of color and depth states
            features[best_neighbor]['color'] = (
                features[best_neighbor]['color'] * old_neighbor_area + 
                features[rid]['color'] * rid_area
            ) / new_area
            features[best_neighbor]['depth'] = (
                features[best_neighbor]['depth'] * old_neighbor_area + 
                features[rid]['depth'] * rid_area
            ) / new_area
            features[best_neighbor]['area'] = new_area

            # # [Probe 3] Monitor the new area after merging
            # if new_area >= area_limit: print(f"   ✅ Merged! New Area {new_area:.0f} >= {area_limit} (Reached)")

            # C. Reconstruct adjacency graph (contour rewiring & state synchronization)
            # Iterate over all neighbors of rid, transferring their relationship
            # with rid to best_neighbor
            for nid, shared_len in current_neighbors.items():
                if nid == best_neighbor:
                    # Simply remove best_neighbor's reference to the now-defunct rid
                    if rid in neighbor_map[best_neighbor]:
                        del neighbor_map[best_neighbor][rid]
                    continue
                
                # Ensure neighbor nid is still active
                if nid not in neighbor_map:
                    continue

                # Logical update: best_neighbor inherits rid's boundary with nid.
                # If best_neighbor was already adjacent to nid, accumulate the
                # boundary length; otherwise create a new edge.
                new_len = neighbor_map[best_neighbor].get(nid, 0) + shared_len
                neighbor_map[best_neighbor][nid] = new_len
                neighbor_map[nid][best_neighbor] = new_len
                
                # Completely erase the de-registered rid from neighbor nid's records
                if rid in neighbor_map[nid]:
                    del neighbor_map[nid][rid]
            
            # De-register this fragment from the global graph
            if rid in neighbor_map:
                del neighbor_map[rid]

        except KeyError as e:
            # Diagnostic block: capture the exception and print the current
            # state-machine slice
            print(f"\n❌ [Purifier Debug] Logic conflict detected:")
            print(f"   - Current fragment ID being processed (rid): {rid}")
            print(f"   - Target ID attempted for merge (best_neighbor): {best_neighbor}")
            print(f"   - Is target ID active: {best_neighbor in neighbor_map}")
            print(f"   - Total neighbors of current fragment: {len(current_neighbors)}")
            print(f"   - Error message: {str(e)}")
            raise e

    # =========================================================================
    # 4. Generate Final Result Map (Final Mapping)
    # =========================================================================
    # Flatten the mapping relationships (handle multi-level skip mappings)
    final_lut = np.arange(max(parent_map.keys()) + 1, dtype=np.uint16)
    for rid in sorted(parent_map.keys()):
        curr = rid
        while parent_map[curr] != curr:
            curr = parent_map[curr]
        final_lut[rid] = curr
        
    refined_mask = final_lut[mask]
    
    return refined_mask