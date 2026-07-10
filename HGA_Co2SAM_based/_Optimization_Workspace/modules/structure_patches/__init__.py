"""
structure_patches/__init__.py
Structure Patches Module (Architecture Hot-Patching)
=====================================================

This module implements the **"Architecture Hot-Patching"** mechanism for the Co2SAM project.
It allows users to dynamically modify, replace, or extend the internal components of the
pretrained SAM model via the configuration file (Config), enabling "config-driven" model
structure evolution without modifying the source code of the model definition.

Design Philosophy:
---------------------------
1. **Separation of Intent & Implementation**:
   - `config.py` is responsible for defining "what to modify, with which strategy" (Intent).
   - This module is responsible for defining "how to modify concretely" (Implementation),
     e.g., building layers, initializing weights, etc.

2. **Non-intrusive Extension**:
   - All structural modification logic is encapsulated in independent submodules
     (e.g., `decoder.py`).
   - The `Model` class only needs to call the unified entry point `apply_structure_patches`
     of this module, without needing to be aware of specific patch details.

3. **Router Dispatch**:
   - Adopts a registry pattern, mapping string keys from Config to concrete handler functions.

Extension Guide:
-------------------------
If a new modification strategy needs to be added (e.g., adding an Adapter to the
Image Encoder):
1. Create a new file in the `structure_patches/` directory (e.g., `encoder.py`).
2. Write the handler function in that file (e.g., `process_adapter(model, strategy)`).
3. Register the new Key-Value mapping in the `PATCH_ROUTERS` dictionary within this
   file (`__init__.py`).
"""

from typing import Dict, Any
import torch.nn as nn

# Import specific handlers from submodules
# Note: the decoder module will be created in the next step
try:
    from .decoder import process_upscaling
except ImportError:
    # Tolerate import failure only during development to avoid errors
    # when viewing documentation
    process_upscaling = None


# =============================================================================
# [Registry] Patch Router Registry
# -----------------------------------------------------------------------------
# This is a core mapping table used to route "logical locators" from Config
# to "concrete handler functions".
#
# Format:
#   Key:   Configuration key in Config (describes the modification target,
#          e.g., 'mask_decoder_upscaling')
#   Value: Handler function (Signature: func(model, strategy_value))
# =============================================================================
PATCH_ROUTERS = {
    # Target: Upsampling module of the Mask Decoder (Output Upscaling)
    # Responsibility: Handles operations such as replacing transposed convolution
    #                 with Resize-Convolution
    'mask_decoder_upscaling': process_upscaling,
}


def apply_structure_patches(model: nn.Module, patches_config: Dict[str, Any]) -> None:
    """
    Unified entry function: apply all structure patches according to the configuration.

    This function acts as a "dispatcher". It iterates over each item in
    `patches_config`, looks up the corresponding handler in the `PATCH_ROUTERS`
    registry, and hands the model instance over to the handler for in-place modification.

    Args:
        model (nn.Module): The SAM model instance to be modified (typically
                           `self.model` under the Co2SAM wrapper).
                           This instance will be passed to specific patch functions
                           for in-place modification.
        patches_config (Dict[str, Any]): Configuration dictionary from
                                         `cfg.model.patches`.
            - Key: Corresponds to a key in `PATCH_ROUTERS`, specifying the target
                   location to modify.
            - Value: Strategy identifier passed to the handler function
                     (e.g., 'resize_convolution').
                     If the Value is None or True, this patch is skipped.

    Returns:
        None. (Modifications are performed in-place.)

    Raises:
        ValueError: If a Key from Config has no corresponding handler in the registry.
        ImportError: If the corresponding handler function is unavailable due to
                     import failure.
    """
    if not patches_config:
        return

    print(f"\n[Structure Patch] Starting architecture patching process...")
    
    applied_count = 0
    for target_key, strategy_value in patches_config.items():
        # 1. Basic validation: skip disabled patches
        if strategy_value is None or strategy_value is True:
            continue

        # 2. Route lookup: obtain the corresponding handler function
        router = PATCH_ROUTERS.get(target_key)
        
        # 3. Integrity check
        if router is None:
            print(f"[Error] [Structure Patch] No router found for key: '{target_key}'. "
                  f"Please check 'PATCH_ROUTERS' in structure_patches/__init__.py.")
            raise ValueError(f"Non-existent target_key: {target_key}")
            
        # 4. Execute dispatch
        print(f"   >> Applying patch to [{target_key}]: Strategy = '{strategy_value}'")
        try:
            # Hand over model control to the concrete implementation module
            router(model, strategy_value)
            applied_count += 1
        except Exception as e:
            print(f"[Error] Failed to apply patch '{target_key}' with strategy '{strategy_value}'.")
            raise e

    print(f"[Structure Patch] Completed. {applied_count} patches applied.\n")