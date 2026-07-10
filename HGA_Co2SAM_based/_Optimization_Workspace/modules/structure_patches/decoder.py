"""
structure_patches/decoder.py
MaskDecoder Structure Implementation
====================================

This module is the backend implementation component of the `structure_patches`
system, specifically responsible for runtime modification and reconfiguration
of the internal structure of the **MaskDecoder** module.

Functional Scope:
--------------------------
This module encapsulates various "surgical" modification logic targeting the SAM
Mask Decoder. It does not involve route dispatching; instead, it focuses on
executing concrete PyTorch module replacement, parameter resetting, layer
addition/removal, and other low-level operations. All handler functions are designed
to receive a `model` instance and a `strategy` parameter, performing in-place
modification on the model.

Usage Context:
-----------------------
Functions in this module are typically referenced by the registry in `__init__.py`
and dispatched uniformly via `model.apply_structure_patches`. Direct invocation from
business logic is not recommended.

Supported Modification Areas:
---------------------------------
1. **Output Upscaling**: Structural replacement of network layers responsible
   for mask upsampling (e.g., TransposeConv -> Resize-Convolution).
"""

import torch.nn as nn
from segment_anything.modeling.common import LayerNorm2d

def process_upscaling(model: nn.Module, strategy: str) -> None:
    """
    Handle structural changes to the upsampling module (output_upscaling) of
    the MaskDecoder.

    This function acts as a modifier for `mask_decoder.output_upscaling`.
    Depending on the `strategy` parameter, it decides whether to replace the
    original transposed convolution layers.

    Args:
        model (nn.Module): SAM model instance (Root Object).
                           This function accesses `model.mask_decoder` for
                           modification.
        strategy (str): Strategy identifier.
            - 'resize_convolution': Replace original layers with
              bilinear upsampling + 3x3 convolution.
            - None or False: Keep the original configuration, no changes (No-op).

    Raises:
        ValueError: Raised when `strategy` is a non-allowed string or type mismatch.
                    (Strict mode: undefined strategy names are not allowed)
    """
    # =========================================================================
    # 1. Strict Parameter Validation
    # =========================================================================
    ALLOWED_STRATEGIES = {
        'resize_convolution', 
    }
    
    # Case A: Default behavior (keep original)
    if strategy is None or strategy is False:
        return

    # Case B: Invalid strategy
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(
            f"[Structure Patch] Invalid strategy '{strategy}' for "
            f"'mask_decoder_upscaling'.\n"
            f"Allowed values are: {ALLOWED_STRATEGIES} or None/False."
        )

    # =========================================================================
    # 2. Execute Strategy: Resize-Convolution
    # =========================================================================
    if strategy == 'resize_convolution':
        _apply_resize_convolution(model)


def _apply_resize_convolution(model: nn.Module) -> None:
    """
    [Internal] Execute the concrete Resize-Convolution replacement logic.
    
    Algorithm Change:
    Old: ConvTranspose2d(k=2, s=2) -> LayerNorm -> GELU ->
         ConvTranspose2d(k=2, s=2) -> GELU
    New: [Upsample(2x) + Conv2d(k=3, p=1)] -> LayerNorm -> GELU ->
         [Upsample(2x) + Conv2d(k=3, p=1)] -> GELU
    """
    mask_decoder = model.mask_decoder
    
    # Obtain the necessary dimensional information (typically 256)
    transformer_dim = mask_decoder.transformer_dim
    
    # Define intermediate channel counts (following the original MaskDecoder logic)
    # Layer 1: dim -> dim/4
    # Layer 2: dim/4 -> dim/8
    dim_layer1_out = transformer_dim // 4
    dim_layer2_out = transformer_dim // 8

    print(
        f"   >> [Action] Replacing 'output_upscaling' with "
        f"Resize-Convolution blocks..."
    )

    # Build the new layer sequence
    new_upscaling = nn.Sequential(
        # --- Block 1 ---
        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
        nn.Conv2d(
            transformer_dim, dim_layer1_out,
            kernel_size=3, stride=1, padding=1
        ),
        LayerNorm2d(dim_layer1_out),
        nn.GELU(),
        
        # --- Block 2 ---
        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
        nn.Conv2d(
            dim_layer1_out, dim_layer2_out,
            kernel_size=3, stride=1, padding=1
        ),
        nn.GELU(),
    )
    
    # Weight Initialization
    # Apply Kaiming Normal only to the newly added Conv2d layers;
    # LayerNorm keeps its default initialization
    for name, module in new_upscaling.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu'
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # Device & Dtype Synchronization
    ref_param = next(mask_decoder.parameters())
    new_upscaling.to(device=ref_param.device, dtype=ref_param.dtype)

    # Execute replacement (In-place Replacement)
    # Assign the newly built module to the original model attribute
    mask_decoder.output_upscaling = new_upscaling

    # [Verification] Explicitly ensure new layers participate in training
    # Although nn.Module defaults to requires_grad=True, explicitly set it
    # for safety and print a log for user auditing
    trainable_params = 0
    for param in new_upscaling.parameters():
        param.requires_grad = True
        trainable_params += param.numel()
    
    print(
        f"   >> [Audit] New layers set to requires_grad=True. "
        f"(Params: {trainable_params})"
    )
    
    print(f"   >> [Success] 'output_upscaling' replaced and re-initialized.")

