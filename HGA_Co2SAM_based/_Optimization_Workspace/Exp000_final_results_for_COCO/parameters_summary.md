Here is the finalized parameter configuration summary file designed for the `Exp000_final_results_for_COCO/` directory. 

Since all COCO experiments utilize the native CT (transposed convolution) decoder, they are all conducted in a **single stage** (trained from scratch), significantly simplifying the execution flow.

---

# `parameters_summary.md`

This document outlines the precise configuration modifications required within `config.py` to reproduce the benchmark runs on the MS COCO 2014 validation set.

Please note that path variables (such as `data_root`, `seg_root`, `namlab_pt_root`, `depth_npy_root`, and checkpoint paths `ckpt`/`stu_ckpt`) are environment-dependent and must be configured to point to your local directories. All algorithmic and optimization hyperparameters listed below should be preserved exactly as shown.

---

## 📂 Quick-Reference Configuration Matrix

| Case | Configuration Name | Backbone | Decoder | HGA Priors | Execution Stage Flow |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | `ViT-B + CT` (Baseline COCO) | ViT-B | CT (Native) | Disabled | 1-Stage (From Scratch) |
| **2** | `ViT-B + CT + HGA` | ViT-B | CT (Native) | Enabled | 1-Stage (From Scratch) |
| **3** | `Eff-XL0 + CT` (Baseline COCO) | Eff-XL0 | CT (Native) | Disabled | 1-Stage (From Scratch) |
| **4** | `Eff-XL0 + CT + HGA` | Eff-XL0 | CT (Native) | Enabled | 1-Stage (From Scratch) 

*\*Note: No RC (Resize-Convolution) decoder experiments were performed on MS COCO 2014; all models are trained and evaluated using the native CT decoder.*

---

## ⚙️ Shared Optimization Baseline

All configurations share baseline values, which are overridden selectively per experiment:

```python
# Shared default loss weights (unless HGA is enabled)
"loss": {
    "weights": {
        "focal": 10.0,
        "dice": 1.0,
        "template": 1.0,
        "contrast": 1.0,
    }
}
```

---

## 🚀 Case-by-Case Configuration Blocks

### Case 1: `ViT-B + CT` (Baseline COCO)
* **Goal**: Single-stage baseline training of ViT-B on COCO with standard transposed convolution.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [200000, 280000],   # Scaled for COCO dataset size
    },
    "model": {
        "type": "vit_b",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {},               # 'mask_decoder_upscaling' is commented out
    },
    "priors": { "enabled": False },
    "output_dir_name": "[vit-b][coco_ct_baseline]",
}
```

---

### Case 2: `ViT-B + CT + HGA` (COCO)
* **Goal**: Single-stage training of the CT model from scratch on COCO with HGA constraints activated.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [200000, 280000],
    },
    "model": {
        "type": "vit_b",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {},
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 0.5,   # Reduced coefficients for COCO
                "logits_1024_depth": 0.5
            },
            "boundary_thinning": {
                "logits_1024": 0.1,          # Reduced coefficient for COCO
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[vit-b][coco_ct_hga]",
}
```

---

### Case 3: `Eff-XL0 + CT` (Baseline COCO)
* **Goal**: Single-stage baseline training of EfficientViT-SAM-XL0 from scratch on COCO with native CT upsampling.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "warmup_steps": 500,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 82783 * 2,              # Equivalent to 2 COCO epochs
        },
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {},
    },
    "priors": { "enabled": False },
    "output_dir_name": "[eff_xl0][coco_ct_baseline]",
}
```

---

### Case 4: `Eff-XL0 + CT + HGA` (COCO)
* **Goal**: Single-stage training of the EfficientViT CT model from scratch on COCO with HGA constraints activated.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "warmup_steps": 500,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 82783 * 2,
        },
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {},
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 0.5,   # Reduced coefficients for COCO
                "logits_1024_depth": 0.5
            },
            "boundary_thinning": {
                "logits_1024": 0.1,          # Reduced coefficient for COCO
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[eff_xl0][coco_ct_hga]",
}
```