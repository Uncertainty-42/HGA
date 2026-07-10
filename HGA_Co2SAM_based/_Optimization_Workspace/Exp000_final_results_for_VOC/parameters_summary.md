This document outlines the precise configuration modifications required within `config.py` to reproduce the 8 benchmark runs on the PASCAL VOC 2012 validation set. 

Please note that path variables (such as `data_root`, `seg_root`, `namlab_pt_root`, `depth_npy_root`, and checkpoint paths `ckpt`/`stu_ckpt`) are environment-dependent and must be configured to point to your local directories. All algorithmic and optimization hyperparameters listed below should be preserved exactly as shown.

---

## 📂 Quick-Reference Configuration Matrix

| Case | Configuration Name | Backbone | Decoder | HGA Priors | Execution Stage Flow |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | `ViT-B + CT` (Baseline) | ViT-B | CT (Native) | Disabled | 1-Stage (From Scratch) |
| **2** | `ViT-B + RC` (Baseline) | ViT-B | RC (Patched) | Disabled | 2-Stage (Stage 1 Warmup $\rightarrow$ Stage 2 Baseline) |
| **3** | `ViT-B + CT + HGA` | ViT-B | CT (Native) | Enabled | 2-Stage (Stage 1 CT Baseline $\rightarrow$ Stage 2 HGA) |
| **4** | `ViT-B + RC + HGA` | ViT-B | RC (Patched) | Enabled | 3-Stage (Stage 1 Warmup $\rightarrow$ Stage 2 Baseline $\rightarrow$ Stage 3 HGA) |
| **5** | `Eff-XL0 + CT` (Baseline) | Eff-XL0 | CT (Native) | Disabled | 1-Stage (From Scratch) |
| **6** | `Eff-XL0 + RC` (Baseline) | Eff-XL0 | RC (Patched) | Disabled | 2-Stage (Stage 1 Warmup $\rightarrow$ Stage 2 Baseline) |
| **7** | `Eff-XL0 + CT + HGA` | Eff-XL0 | CT (Native) | Enabled | 1-Stage (From Scratch) |
| **8** | `Eff-XL0 + RC + HGA` | Eff-XL0 | RC (Patched) | Enabled | 3-Stage (Stage 1 Warmup $\rightarrow$ Stage 2 Baseline $\rightarrow$ Stage 3 HGA) |

---

## ⚙️ Shared Optimization Baseline

All configurations share baseline values, which are overridden selectively per experiment:

```python
# Shared default loss weights (unless modified per stage)
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

### Case 1: `ViT-B + CT` (Baseline)
* **Goal**: Single-stage baseline training of ViT-B with standard transposed convolution.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [30000, 60000],
    },
    "model": {
        "type": "vit_b",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {}, # 'mask_decoder_upscaling' is commented out
    },
    "priors": { "enabled": False },
    "output_dir_name": "[vit-b][ct_baseline]",
}
```

---

### Case 2: `ViT-B + RC` (Baseline)
* **Goal**: Multi-stage training to adapt the Resize-Convolution decoder on ViT-B without HGA.

#### ➡️ Stage 1 (RC Warmup)
*Only train the RC patch parameters using Template Loss.*
```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [30000, 60000],
        "dynamic_schemes": {
            "template": {
                "output_upscaling": {
                    "allow_bounce": False,
                    "rates": {0.5: 500, 0.3: 100, 0.2: 10},
                    'window_size': 20
                }
            }
        }
    },
    "model": {
        "type": "vit_b",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 0.0,      # Commented out / Disabled
            "dice": 0.0,       # Commented out / Disabled
            "template": 1.0,   # Active
            "contrast": 0.0,   # Commented out / Disabled
        }
    },
    "priors": { "enabled": False },
    "output_dir_name": "[vit-b][rc_stage1_warmup]",
}
```

#### ➡️ Stage 2 (Baseline Training)
*Resume training using all baseline losses, loading the optimal Stage 1 warmup checkpoint.*
```python
config_override = { "resume": True }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [30000, 60000],
        "dynamic_schemes": {} # Disabled / Commented out
    },
    "model": {
        "type": "vit_b",
        "ckpt": "path/to/stage1/best_u_teacher.pth",
        "stu_ckpt": "path/to/stage1/best_student.pth",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
        }
    },
    "priors": { "enabled": False },
    "output_dir_name": "[vit-b][rc_stage2_baseline]",
}
```

---

### Case 3: `ViT-B + CT + HGA`
* **Goal**: Two-stage training of the CT model, continuing directly from the converged CT baseline.

#### ➡️ Stage 1 (CT Baseline)
*Refer to **Case 1** and train until convergence.*

#### ➡️ Stage 2 (HGA Fine-Tuning)
*Load the optimal CT baseline checkpoints, and activate the HGA constraints.*
```python
config_override = { "resume": True }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [30000, 60000],
    },
    "model": {
        "type": "vit_b",
        "ckpt": "path/to/case1/best_u_teacher.pth",
        "stu_ckpt": "path/to/case1/best_student.pth",
        "patches": {}, # 'mask_decoder_upscaling' is commented out
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 1.0,
                "logits_1024_depth": 1.0
            },
            "boundary_thinning": {
                "logits_1024": 1.0,
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[vit-b][ct_stage2_hga]",
}
```

---

### Case 4: `ViT-B + RC + HGA`
* **Goal**: Three-stage training of the RC model, injecting HGA constraints onto the converged RC baseline.

#### ➡️ Stage 1 (RC Warmup)
*Refer to **Case 2 - Stage 1** and train until completion.*

#### ➡️ Stage 2 (RC Baseline)
*Refer to **Case 2 - Stage 2** and train until convergence.*

#### ➡️ Stage 3 (HGA Fine-Tuning)
*Load the optimal Case 2 Stage 2 checkpoints, and activate HGA constraints.*
```python
config_override = { "resume": True }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "step",
        "steps": [30000, 60000],
        "dynamic_schemes": {}
    },
    "model": {
        "type": "vit_b",
        "ckpt": "path/to/case2_stage2/best_u_teacher.pth",
        "stu_ckpt": "path/to/case2_stage2/best_student.pth",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 1.0,
                "logits_1024_depth": 1.0
            },
            "boundary_thinning": {
                "logits_1024": 1.0,
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[vit-b][rc_stage3_hga]",
}
```

---

### Case 5: `Eff-XL0 + CT` (Baseline)
* **Goal**: Single-stage baseline training of EfficientViT-SAM-XL0 from scratch with native CT upsampling.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "warmup_steps": 500,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 10582 * 4,
        },
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {}, # 'mask_decoder_upscaling' is commented out
    },
    "priors": { "enabled": False },
    "output_dir_name": "[eff_xl0][ct_baseline]",
}
```

---

### Case 6: `Eff-XL0 + RC` (Baseline)
* **Goal**: Two-stage training to adapt the RC decoder on the upgraded EfficientViT backbone.

#### ➡️ Stage 1 (RC Warmup)
```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 10582 * 2,
        },
        "dynamic_schemes": {
            "template": {
                "output_upscaling": {
                    "allow_bounce": False,
                    "rates": {0.5: 500, 0.35: 300, 0.2: 100, 0.15: 10, 0.1: 5},
                    'window_size': 20
                }
            }
        }
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 0.0,
            "dice": 0.0,
            "template": 1.0,
            "contrast": 0.0,
        }
    },
    "priors": { "enabled": False },
    "output_dir_name": "[eff_xl0][rc_stage1_warmup]",
}
```

#### ➡️ Stage 2 (Baseline Training)
```python
config_override = { "resume": True }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 10582 * 2,
        },
        "dynamic_schemes": {}
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "path/to/stage1/best_u_teacher.pth",
        "stu_ckpt": "path/to/stage1/best_student.pth",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
        }
    },
    "priors": { "enabled": False },
    "output_dir_name": "[eff_xl0][rc_stage2_baseline]",
}
```

---

### Case 7: `Eff-XL0 + CT + HGA`
* **Goal**: Single-stage training from scratch, directly injecting HGA constraints on the CT path.

```python
config_override = { "resume": False }

base_config = {
    "opt": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "warmup_steps": 500,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 10582 * 4,
        },
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "",
        "stu_ckpt": "",
        "patches": {}, # 'mask_decoder_upscaling' is commented out
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 1.0,
                "logits_1024_depth": 1.0
            },
            "boundary_thinning": {
                "logits_1024": 1.0,
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[eff_xl0][ct_hga]",
}
```

---

### Case 8: `Eff-XL0 + RC + HGA`
* **Goal**: Three-stage training of the RC model, injecting HGA constraints onto the converged RC baseline.

#### ➡️ Stage 1 (RC Warmup)
*Refer to **Case 6 - Stage 1** and train until completion.*

#### ➡️ Stage 2 (RC Baseline)
*Refer to **Case 6 - Stage 2** and train until convergence.*

#### ➡️ Stage 3 (HGA Fine-Tuning)
*Load the optimal Case 6 Stage 2 checkpoints, and activate HGA constraints.*
```python
config_override = { "resume": True }

base_config = {
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 250,
        "schedule_type": "gaussian",
        "gaussian": {
            "sigma": 10582 * 2,
        },
        "dynamic_schemes": {}
    },
    "model": {
        "type": "eff_xl0",
        "ckpt": "path/to/case6_stage2/best_u_teacher.pth",
        "stu_ckpt": "path/to/case6_stage2/best_student.pth",
        "patches": {
            'mask_decoder_upscaling': 'resize_convolution',
        },
    },
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,
            "top2_edge_alignment": {
                "logits_1024_namlab": 1.0,
                "logits_1024_depth": 1.0
            },
            "boundary_thinning": {
                "logits_1024": 1.0,
            }
        }
    },
    "priors": { "enabled": True },
    "output_dir_name": "[eff_xl0][rc_stage3_hga]",
}
```