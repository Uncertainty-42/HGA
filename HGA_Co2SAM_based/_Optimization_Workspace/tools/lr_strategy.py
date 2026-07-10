"""
Dynamic LR Strategy Management System
=====================================

This module handles the dynamic learning rate control logic based on "architectural regex matching"
for research experiments. It solves the problem of precisely locating atomic layers and adjusting
learning rate multipliers based on real-time loss feedback, under the context of model structural
changes (e.g., Patches).

Core Philosophy:
-------------------------
1. **Deterministic Localization**: Reject fuzzy string matching; mandatory full-sequence comparison
   against "structural fingerprints".
2. **Fail-Fast**: Immediately abort on architecture mismatches, configuration conflicts, or policy
   violations. No ambiguity is tolerated.
3. **Atomic Control**: Only the selected layers are accelerated; unselected layers in the sequence
   remain at their original rate.

Design Specification:
------------------------------
- **Sliding Window**: Default 20-iteration window period (customizable via `window_size` in config).
  No adjustments are triggered until the window is full.
- **Unidirectional Locking**: Supports `allow_bounce` configuration; rebound of the multiplier is
  disabled by default.
- **Stacked Multiplication**: The dynamic multiplier stacks multiplicatively with the global
  scheduler (Warmup/Decay).
"""

import torch
import torch.nn as nn
from collections import deque
from segment_anything.modeling.common import LayerNorm2d

# =============================================================================
# Architectural Regex Registry
# =============================================================================
# Format specification:
# Key: Logical locator key, corresponding to key names in config.py.
# Value: {
#   'path': str, hierarchical access path to the target module in the model
#           (e.g., 'mask_decoder.output_upscaling').
#   'pattern': List[Dict], defines the full-sequence fingerprint.
#              Must include 'type' (layer type) and 'attrs' (attribute check).
#   'target_indices': List[int], specifies which sub-module parameters in the
#                     sequence need to be captured.
# }
# =============================================================================
DYNAMIC_LR_REGISTRY = {
    'output_upscaling': {
        'path': 'mask_decoder.output_upscaling',  # Logical path from model root to target module
        'pattern': [
            {'type': nn.Upsample, 'attrs': {'scale_factor': 2.0}},
            {'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'type': (nn.LayerNorm, LayerNorm2d), 'attrs': {}},
            {'type': (nn.GELU, nn.ReLU), 'attrs': {}},
            {'type': nn.Upsample, 'attrs': {'scale_factor': 2.0}},
            {'type': nn.Conv2d, 'attrs': {'kernel_size': (3, 3)}},
            {'type': (nn.GELU, nn.ReLU), 'attrs': {}},
        ],
        'target_indices': [1, 5],
    }
}

class LRStrategyManager:
    """
    Dynamic learning rate phase manager.

    Responsible for maintaining the learning rate evolution state of specific parameter groups.
    It monitors the moving average of a specific loss, checks against the configured stepwise
    thresholds (rates), and modifies the optimizer's param_groups in real-time while respecting
    the "window period" and "unidirectionality" constraints.

    Attributes:
        history (deque): Sliding window of length window_size that stores historical loss values.
        last_applied_multiplier (float): The currently effective multiplier weight, used to
            reduce unnecessary optimizer writes.
        min_multiplier_achieved (float): The lowest multiplier reached historically, used to
            implement the unidirectional locking logic.
    """
    def __init__(self, group_idx, loss_key, scheme_cfg):
        """
        Initialize the dynamic learning rate manager.

        Args:
            group_idx (int): Index position of the target parameter group in the optimizer.
            loss_key (str): Reference loss key name to monitor (e.g., 'template').
            scheme_cfg (Box/dict): Specific strategy configuration.
                - rates (dict): Mapping from thresholds to multipliers, e.g., {0.5: 50, 0.2: 10}.
                - allow_bounce (bool, optional): Whether rebound of the multiplier is allowed. Defaults to False.
                - window_size (int, optional): Sliding window size for the moving average. Defaults to 20.
        """
        self.group_idx = group_idx
        self.loss_key = loss_key
        self.allow_bounce = scheme_cfg.get('allow_bounce', False)
        
        # [Config-driven window] Retrieve window size; defaults to 20
        self.window_size = scheme_cfg.get('window_size', 20)
        
        # Sort rates as descending thresholds to ensure correct stepwise retrieval logic
        self.rates = sorted(scheme_cfg['rates'].items(), key=lambda x: x[0], reverse=True)
        
        # Initialize history deque with the configured window size
        self.history = deque(maxlen=self.window_size)
        
        self.min_multiplier_achieved = float('inf')
        self.last_applied_multiplier = 1.0

        self.window_filled_notified = False  # Audit flag: ensures evidence is printed only once when window fills

    def update(self, loss_dict, optimizer):
        """
        Update the multiplier based on the loss state of the current iteration.

        Execution flow:
        1. Collect Loss: Extract the value corresponding to loss_key from loss_dict and push it into the deque.
        2. Gate Check: If the deque length has not reached window_size, forcibly maintain the 1.0 multiplier
           and do not execute any logic.
        3. Threshold Matching: Once the deque is full, compute the moving average (MA), and match the
           theoretical multiplier against the descending thresholds.
        4. Unidirectional Locking: If allow_bounce is False, the new multiplier must not exceed the
           historically achieved minimum.
        5. Physical Application: If the multiplier transitions to a new stage, directly modify the
           corresponding optimizer param_group['lr'].

        Args:
            loss_dict (dict): Dictionary containing sub-losses, returned by Co2SAMCriterion.
            optimizer (torch.optim.Optimizer): The optimizer instance from the host environment.
        """
        val = loss_dict.get(self.loss_key)
        if val is None: return
        
        # Handle Tensor type
        if isinstance(val, torch.Tensor):
            val = val.item()
            
        self.history.append(val)
        
        assert self.history.maxlen is not None, "[ERROR] self.history.maxlen is None"
        # Window not full — force 1.0
        if len(self.history) < self.history.maxlen:
            return

        ma = sum(self.history) / self.history.maxlen
        
        # 1. Stepwise retrieval logic (unified use of "greater-than" check)
        multiplier = 1.0
        for threshold, m in self.rates:
            if ma > threshold:
                multiplier = m
                break
        
        # 2. Apply "unidirectionality" state locking
        if not self.allow_bounce:
            multiplier = min(multiplier, self.min_multiplier_achieved)
        self.min_multiplier_achieved = min(multiplier, self.min_multiplier_achieved)

        # 3. Force-apply the multiplier every iteration to counter Scheduler's reset override
        # At this point, param_groups[idx]['lr'] has already been updated by LambdaLR to the base value
        optimizer.param_groups[self.group_idx]['lr'] *= multiplier

        # 4. [Evidence Audit] On the first iteration when the window becomes full, force-print current state
        if not self.window_filled_notified and len(self.history) == self.history.maxlen:
            actual_lr = optimizer.param_groups[self.group_idx]['lr']
            print(f"[DynamicLR Audit] Window Closed | Key: {self.loss_key} | MA: {ma:.4f} "
                  f"| Applied Multiplier: {multiplier}x | Real-Time LR: {actual_lr:.8f}")
            self.window_filled_notified = True

        # 5. Only print change log when the multiplier stage transitions
        if multiplier != self.last_applied_multiplier:
            actual_lr = optimizer.param_groups[self.group_idx]['lr']
            self.last_applied_multiplier = multiplier
            print(f"[DynamicLR] Stage Change | {self.loss_key} -> {multiplier}x "
                  f"| New Effective LR: {actual_lr:.8f}")

def resolve_parameters_by_regex(model, locator_key):
    """
    Architectural regex-based parameter resolution auditor.

    This function performs "full-sequence fingerprint matching." It not only checks the target
    attribute names, but also deeply verifies that the type and key attributes of each layer
    inside the entire nn.Sequential match the pattern registered in the registry exactly.

    Args:
        model (nn.Module): Root model instance to audit.
        locator_key (str): Logical key name in the registry.

    Returns:
        List[nn.Parameter]: The list of atomic-layer parameters captured according to
            target_indices after a successful match.

    Raises:
        RuntimeError: Raised when the sequence length, layer type, or attributes do not match
            the expected fingerprint. Execution is refused.
        ValueError: Raised when the locator key is not defined in the registry.
    """
    if locator_key not in DYNAMIC_LR_REGISTRY:
        raise ValueError(f"Unknown locator key: {locator_key}")
    
    rule = DYNAMIC_LR_REGISTRY[locator_key]

    # Dynamically resolve path: starting from the model root object, chain-retrieve
    # the target module according to the 'path' defined in the registry
    target_obj = model
    path_segments = rule['path'].split('.')
    for segment in path_segments:
        if not hasattr(target_obj, segment):
            raise RuntimeError(
                f"Structural Audit Failed: Model path segment '{segment}' not found "
                f"while resolving '{rule['path']}' for key '{locator_key}'."
            )
        target_obj = getattr(target_obj, segment)
    
    seq = target_obj

    if not isinstance(seq, nn.Sequential):
        raise RuntimeError(f"Layer {locator_key} is not nn.Sequential, cannot match regex.")

    # 1. Full-sequence matching
    if len(seq) != len(rule['pattern']):
        raise RuntimeError(f"Architecture mismatch for {locator_key}. Expected len {len(rule['pattern'])}, got {len(seq)}")

    for i, (module, p) in enumerate(zip(seq, rule['pattern'])):
        # Type check
        if not isinstance(module, p['type']):
            raise RuntimeError(f"Index {i} mismatch: Expected {p['type']}, got {type(module)}")
        # Attribute check (e.g., scale_factor, kernel_size)
        for attr, val in p['attrs'].items():
            actual_val = getattr(module, attr, None)
            if actual_val != val:
                raise RuntimeError(f"Index {i} attr mismatch: Expected {attr}={val}, got {actual_val}")

    # 2. Capture target parameters
    params = []
    for idx in rule['target_indices']:
        params.extend(list(seq[idx].parameters()))
    
    return params