import torch
import torch.nn as nn
import math
from sam_lora import LoRA  # Inherit from the original base class to reuse parameter save/load logic

class _LoRA_qkv_conv(nn.Module):
    """
    EfficientViT-SAM QKV convolution layer LoRA wrapper class.

    This class injects a low-rank decomposition (Low-Rank Adaptation) path in parallel
    beside the original 1x1 convolution operator, enabling parameter-efficient fine-tuning
    of the Query (Q) and Value (V) projection weights within the attention mechanism.

    Core mechanism:
        1. 4D tensor compatibility: Unlike the original SAM which processes 3D [B, N, C]
           sequences, this class is specifically designed for 4D [B, C, H, W] feature maps
           in convolutional architectures.
        2. Targeted slicing injection: EfficientViT concatenates Q/K/V results along the
           output channel dimension. This class uses index-based slicing to add LoRA
           deltas only to Q (first 1/3 of channels) and V (last 1/3 of channels), ensuring
           the Key (K) branch remains fully frozen, strictly aligning with the original
           SAM-LoRA logic.

    Args:
        qkv (nn.Module): Original pre-trained 1x1 convolution layer instance.
        linear_a_q (nn.Module): Down-projection LoRA convolution matrix for the Query branch (1x1 Conv).
        linear_b_q (nn.Module): Up-projection LoRA convolution matrix for the Query branch (1x1 Conv).
        linear_a_v (nn.Module): Down-projection LoRA convolution matrix for the Value branch (1x1 Conv).
        linear_b_v (nn.Module): Up-projection LoRA convolution matrix for the Value branch (1x1 Conv).
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the original convolution output and add the low-rank
        deltas from the Q/V branches.

        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            torch.Tensor: LoRA-corrected QKV feature map of shape [B, 3C, H, W].
        """
        # x: [B, C, H, W]
        qkv = self.qkv(x)  # [B, 3C, H, W]
        new_q = self.linear_b_q(self.linear_a_q(x)) # [B, C, H, W]
        new_v = self.linear_b_v(self.linear_a_v(x)) # [B, C, H, W]
        
        # Slice-based accumulation on 4D tensors along Dim 1 (Channel)
        qkv[:, : self.dim, :, :] += new_q
        qkv[:, -self.dim :, :, :] += new_v
        return qkv
    
class _LoRA_point_conv(nn.Module):
    """
    EfficientViT-SAM Neck bottleneck layer LoRA wrapper class.

    This class fine-tunes the 1x1 point-wise convolution in the Neck structure
    (e.g., regions 3-10). Unlike the QKV structure, this class applies full-channel
    LoRA delta addition on the output channels to optimize multi-scale feature fusion weights.

    Core mechanism:
        1. Cross-channel feature recalibration: Adjusts inter-channel feature interaction
           logic via the low-rank bypass without altering the original spatial receptive field.
        2. Full additive modification: The output dimension of the LoRA path matches that
           of the original convolution, enabling direct element-wise addition.

    Args:
        point_conv (nn.Module): Original pre-trained 1x1 point-wise convolution layer instance
            (typically In_C=1024, Out_C=256).
        linear_a (nn.Module): Down-projection LoRA convolution matrix (1x1 Conv, 1024 -> r).
        linear_b (nn.Module): Up-projection LoRA convolution matrix (1x1 Conv, r -> 256).
    """
    def __init__(
        self,
        point_conv: nn.Module,
        linear_a: nn.Module,
        linear_b: nn.Module,
    ):
        super().__init__()
        self.point_conv = point_conv
        self.linear_a = linear_a
        self.linear_b = linear_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the original point-wise convolution output and add the
        full-channel low-rank delta.

        Args:
            x (torch.Tensor): Input feature map of shape [B, In_C, H, W].

        Returns:
            torch.Tensor: LoRA-corrected feature map of shape [B, Out_C, H, W].
        """
        # Direct summation of the original path and the LoRA path
        return self.point_conv(x) + self.linear_b(self.linear_a(x))
    
class LoRA_EffSam(LoRA):
    """
    Main class for EfficientViT-SAM LoRA injection management.

    This class parses the global configuration file and dynamically injects
    convolutional LoRA operators into Stage 4/5 of the EfficientViT backbone and
    the Neck structure according to the specified enable states and ranks.

    Core capabilities:
        1. Automated Surgery: Leverages recursive path probing to precisely replace
           deeply nested, complexly named modules.
        2. Multi-Rank Adaptation: Supports configuring differentiated LoRA ranks for
           different hierarchy levels (Stage 4, 5, Neck).
        3. Gradient Management: Automatically freezes all parameters of the image encoder,
           allowing gradient updates only for LoRA bypass weights.

    Args:
        sam_model (Sam): EfficientViT-SAM model instance to be fine-tuned.
        cfg (Box): Global configuration object containing the lora_settings branch.
    """

    def __init__(self, sam_model: nn.Module, cfg):
        super(LoRA_EffSam, self).__init__()

        # 1. Extract configuration reference
        self.config = cfg.model.lora_settings.eff_sam.target_modules
        self.lora_vit = sam_model  # Keep attribute name consistent with the base class
        
        # 2. Initialize storage containers (for base class method save_lora_parameters to iterate)
        self.w_As = [] 
        self.w_Bs = []

        # 3. Freeze all parameters of the image encoder
        for param in self.lora_vit.image_encoder.parameters():
            param.requires_grad = False

        # 4. Perform injection surgery
        self._inject_lora_modules()
        
        # 5. Initialize newly added parameters
        self.reset_parameters()

    def _inject_lora_modules(self):
        """
        Probe and replace target modules according to configuration.
        """
        # Iterate over all named sub-modules (note: must precisely match mit-han-lab's internal naming convention)
        for name, module in self.lora_vit.image_encoder.named_modules():
            
            # --- Scenario A: Handle QKV of Stage 4 and Stage 5 ---
            if "context_module.main.qkv.conv" in name:
                stage_key = "stage4" if "stages.4" in name else "stage5" if "stages.5" in name else None
                
                if stage_key and self.config[stage_key].enabled:
                    r = self.config[stage_key].rank
                    self._replace_qkv_module(name, module, r)

            # --- Scenario B: Handle Point Conv of Neck (3-10) ---
            elif "neck.middle.op_list" in name and name.endswith("point_conv.conv"):
                if self.config.neck.enabled:
                    r = self.config.neck.rank
                    self._replace_point_module(name, module, r)

    def _replace_qkv_module(self, name: str, original_conv: nn.Module, r: int):
        """
        Perform QKV convolution layer replacement.

        This method locates the parent module of the target operator via reflection
        and replaces it with the _LoRA_qkv_conv wrapper class.

        Args:
            name (str): Full dotted path of the operator.
            original_conv (nn.Module): Original 1x1 convolution module.
            r (int): LoRA rank.
        """
        in_c = original_conv.in_channels
        
        # Instantiate two low-rank bypass pathways for Q/V (1x1 convolution)
        a_q = nn.Conv2d(in_c, r, 1, bias=False)
        b_q = nn.Conv2d(r, in_c, 1, bias=False)
        a_v = nn.Conv2d(in_c, r, 1, bias=False)
        b_v = nn.Conv2d(r, in_c, 1, bias=False)

        # Register to lists for base-class save/load of weights
        self.w_As.extend([a_q, a_v])
        self.w_Bs.extend([b_q, b_v])

        # Build the wrapper class
        wrapper = _LoRA_qkv_conv(original_conv, a_q, b_q, a_v, b_v)
        
        # Locate the parent module and perform attribute replacement
        parent_name, _, child_name = name.rpartition('.')
        parent_module = self.lora_vit.image_encoder.get_submodule(parent_name)
        setattr(parent_module, child_name, wrapper)

    def _replace_point_module(self, name: str, original_conv: nn.Module, r: int):
        """
        Perform Neck point-wise convolution layer replacement.

        This method fine-tunes bottleneck layers of non-QKV structure by replacing
        the original point-wise convolution with _LoRA_point_conv.

        Args:
            name (str): Full dotted path of the operator.
            original_conv (nn.Module): Original 1x1 convolution module.
            r (int): LoRA rank.
        """
        in_c = original_conv.in_channels
        out_c = original_conv.out_channels
        
        # Instantiate full-channel low-rank bypass (1x1 convolution)
        a = nn.Conv2d(in_c, r, 1, bias=False)
        b = nn.Conv2d(r, out_c, 1, bias=False)

        # Register parameters
        self.w_As.append(a)
        self.w_Bs.append(b)

        # Build the wrapper class
        wrapper = _LoRA_point_conv(original_conv, a, b)
        
        # Perform attribute replacement
        parent_name, _, child_name = name.rpartition('.')
        parent_module = self.lora_vit.image_encoder.get_submodule(parent_name)
        setattr(parent_module, child_name, wrapper)

    def reset_parameters(self) -> None:
        """
        Initialize all LoRA bypass matrices.

        This method strictly follows the initialization principles from the LoRA paper:
        1. A matrices (Down-projection) are initialized with Kaiming uniform distribution
           to ensure feature extraction layer activity.
        2. B matrices (Up-projection) are initialized to all zeros, ensuring that at the
           start of training the LoRA path output is zero, thereby keeping the fine-tuned
           model fully equivalent to the original pre-trained model in its initial state,
           improving training convergence stability.
        """
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


def fix_bn_states(*models: nn.Module):
    """
    Force-lock the BatchNorm running state of the Image Encoder region in the model.

    This function addresses the problem where, during CNN backbone (e.g., EfficientViT)
    fine-tuning, the recursive call of model.train() causes pre-trained BN statistics
    to be "contaminated" by training data.

    Core mechanism:
        1. State-override defense: During the training loop, PyTorch's model.train()
           switches all sub-modules to Training mode. This function explicitly calls
           .eval() to forcibly pull BN back to Eval mode, thereby ensuring the model uses
           frozen pre-trained running mean and variance.
        2. Multi-model decoupling support: Supports passing multiple model instances
           simultaneously (e.g., training model, teacher model, or template model) for
           unified state management.

    Note: This function only disables BN statistics updates (i.e., mode switching);
    parameter gradient freezing must still be ensured via requires_grad = False.

    Args:
        *models (nn.Module): One or more model instances to process.
    """
    for model in models:
        if model is not None:
            # Prioritize locating the image encoder level to avoid accidentally
            # affecting other regions that need trainable BN (e.g., newly added decoder layers)
            target_submodule = model.image_encoder if hasattr(model, "image_encoder") else model
            
            for m in target_submodule.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    # Key operation: force BN into eval mode, locking statistics buffers
                    m.eval()
