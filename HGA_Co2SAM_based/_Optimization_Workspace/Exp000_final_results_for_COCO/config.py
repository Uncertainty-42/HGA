# config.py
from box import Box
from pathlib import Path
cur_path = Path(__file__)
print(f"⛲⛲⛲⛲⛲⛲ Current config file: {str(cur_path)} ⛲⛲⛲⛲⛲⛲")

base_config = {
    "eval_interval": 1,
    # ⚠️⚠️ True:  Oracle evaluation, i.e., filter out classes not present in
    #            sample GT during evaluation.
    #       False: Fair competition, i.e., no peeking at GT during evaluation;
    #              DINO says what classes exist, the model predicts them.
    #       Default: False, fair competition.
    "eval_with_oracle_filter": False,
    "ema_rate": 0.999,
    "opt": {
        # ⭐⭐ Options: "adam", "adamw". Default if missing: "adam"
        "type": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        # ⭐⭐ Gradient clipping threshold. None or missing means no clipping.
        "clip_grad": None,

        "warmup_steps": 250,

        # ⭐⭐ Options: "step", "poly", "cosine", "gaussian".
        #    Default if missing: "step"
        "schedule_type": "step",

        # --- Step schedule parameters (original parameters unchanged) ---
        "decay_factor": 10,
        "steps": [30000, 60000],

        # === Poly / Cosine / Gaussian schedule parameters === ⭐⭐
        "poly": {
            "max_steps": 10582 * 10,   # Total steps (10 epochs)
            "power": 0.9,              # Poly-specific
            "min_lr_ratio": 0.01       # Minimum LR ratio (e.g., decay to 1%
                                       # of initial value)
        },
        "cosine": {
            "max_steps": 10582 * 10,   # Total steps (10 epochs)
            "min_lr_ratio": 0.01       # Minimum LR ratio
        },
        "gaussian": {
            "sigma": 10582 * 3,       # Inflection point step count
        },
        # === Dynamic learning rate multiplier interface === ⭐⭐
        "dynamic_schemes": {
            # "template": {
            #     "output_upscaling": {
            #         "allow_bounce": False,
            #         "rates": {0.5: 500, 0.35: 300, 0.2: 100, 0.15: 10,
            #                   0.1: 5},
            #         'window_size': 20
            #     }
            # }
            # "template": {
            #     "output_upscaling": {
            #         "allow_bounce": False,
            #         "rates": {0.5: 500, 0.3: 100, 0.2: 10},
            #         'window_size': 20
            #     }
            # }
        }
        # ==========================
    },
    # === Visualization configuration interface === ⭐⭐
    "visual": {
        "enabled": True,             # Master switch
        "eval_enabled": True,        # Whether to visualize during evaluation
        # Probe whitelist (default: empty, no probes activated)
        "probes": [
            "val_alignment_check",              # in eval_utils.py
            'baseline_monitoring',              # in train_voc.py / train_coco.py
            'top2_edge_alignment_monitoring',   # in train_voc.py / train_coco.py
                                                # & losses.py
            'boundary_thinning_monitoring',     # in losses.py
            'aug_pairs',                        # in train_voc.py / train_coco.py
                                                # strong/weak augmented image pairs
            'dino_boxes',                       # in train_voc.py / train_coco.py
                                                # & eval_utils.py DINO detection boxes
            'depth_prior_generation',           # in model.py
            'show_blurred_without_sigma^4_Comp', # in model.py
        ],
        "policy": {
            "warmup_steps": 500,     # First 500 iterations
            "warmup_freq": 20,       # Every 20 iterations
            "epoch1_freq": 200,      # Remaining part of Epoch 1: every 200 iters
            "later_freq": 1000,      # Subsequent epochs: every 1000 iters
        }
    },
    # ==========================
    "model": {
        # Options:
        #   vit_b:  Baseline model, 91M
        #   vit_l:  Larger variant, 308M
        #   vit_h:  Even larger variant, 636M
        #   eff_xl0: EfficientViT-SAM-XL0, 117.0M
        #   eff_xl1: EfficientViT-SAM-XL1, 203.3M
        "type": "vit_b",
        "versions": {
            "vit": ["vit_b", "vit_l", "vit_h"],
            "eff_vit": ["eff_xl0", "eff_xl1"]
        },
        "lora_settings": {
            "eff_sam": {
                "target_modules": {
                    "stage4": {
                        "enabled": True,
                        "rank": 4
                    },
                    "stage5": {
                        "enabled": True,
                        "rank": 4
                    },
                    "neck": {
                        "enabled": True,
                        "rank": 4
                    },
                }
            }
        },
        "eff_mapping": {
            "eff_xl0": "efficientvit-sam-xl0",
            "eff_xl1": "efficientvit-sam-xl1",
        },
        "checkpoint": "./checkpoints/",
        # If using a checkpoint, what is its absolute path? ↓ ⭐⭐
        "ckpt": "",
        "stu_ckpt": "",
        # ⭐⭐ Whether to set all parameters frozen. (Later, precise
        # unfreezing is configured as needed, i.e., freeze all first, then
        # selectively unfreeze. If set to False, keep the default unfreeze
        # strategy and apply precise unfreezing on top of it.)
        "set_all_params_frozen": False,
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
            # "ConvTranspose": False,   # The two transposed convolutions in
            #                           the baseline decoder upsampling.
            #                           Default: True, i.e., keep frozen
            #                           (original code logic).
            # "resize_convolution": True # Freeze the replaced RC module.
            #                           Default: False, i.e., not frozen.
        },
        # === Model layer add/remove/modify configuration interface === ⭐⭐
        # Key:   Logical locator (describes the modification target)
        # Value: Strategy identifier (describes which algorithm to use),
        #        or True to delete, or False/None (default) to skip
        #        (since missing key returns None/False, matching default).
        "patches": {
            # Exp02: resize_convolution (Resize-Convolution,
            #         resolving checkerboard artifacts)
            # 'mask_decoder_upscaling': 'resize_convolution',
        },
        # ==========================
    },
    # === Loss configuration interface === ⭐⭐
    "loss": {
        "weights": {
            "focal": 10.0,
            "dice": 1.0,
            "template": 1.0,
            "contrast": 1.0,

            # Top-2 Decision-Competition Alignment Loss sub-items
            # "top2_edge_alignment": {
            #     "logits_256_namlab": 0.0,
            #     "logits_256_depth": 0.0,
            #     "logits_1024_namlab": 1.0,
            #     "logits_1024_depth": 1.0
            # },

            # Boundary-Thinning Loss: suppresses inflated semantic edges,
            # forcing the model to decisively contract at boundaries
            # "boundary_thinning": {
            #     "logits_256": 0.0,  # Thinning weight at 256 resolution
            #     "logits_1024": 1.0, # Thinning weight at 1024 resolution
            #     "threshold": 10.0   # Local energy threshold (tau):
            #                        #   maximum total confidence within a
            #                        #   5x5 region; exceeding this budget
            #                        #   triggers Hinge Loss penalty
            # }
        }
    },
    # ==========================
    # === Output directory name === ⭐⭐
    "output_dir_name": "[vit-b][baseline_test]",
    # Concatenated as a human-readable note appended to the timestamped
    # output directory name, e.g.:
    # '20251210_222910__[baseline_0.8219_CT_frozen][...]'
    # "[baseline_0.8219_CT_frozen][gaussian_sigma_3.5,3.5]...
    #  [alignment_0,0,2,2][boundary_thinning_0,1,10]",
    # ==========================
    # === Whether to evaluate before training === ⭐⭐
    "eval_before_training": False,
    # ==========================
    # === Hierarchical-Geometric Priors configuration interface === ⭐⭐
    "priors": {
        "enabled": False,
        "namlab": {
            # Stage A: Target merge level for NAMLab sequential merging
            # (e.g., 60)
            "target_level": 60,

            # Whether to enable the purification strategy below. If False,
            # do not purify, directly return the raw NAMLab target level.
            # Default: True.
            "purification_enabled": True,

            # Small region removal strategy (Stage C)
            "area_constraint": {
                # Strategy selection: 'pxl' (absolute pixels)
                #                   or 'ratio' (global proportion)
                "strategy": "pxl",
                # Numeric value corresponding to the chosen strategy
                # (pxl: 4700, ratio: 0.01)
                "threshold": 4700,
            },

            # Computation specification
            "color_logic": {
                "space": "Lab",
                "illuminant": "D50",         # Default D50, alternative: D65
                "lib": "skimage",            # Conversion library: 'skimage'
                                             # or 'opencv'
            },

            # Cost formula parameters (Stage C)
            "cost_params": {
                "epsilon": 1e-6,             # Prevent division by zero
            },

            # Path and cache management (read by priors_manager)
            "paths": {
                # Raw data inputs
                "raw_pt_dir": (
                    "path/to/your/namlab_pt_300L"
                ),
                "raw_depth_dir": (
                    "path/to/your/depth_anything"
                ),

                # Refinement result output (cache root)
                "cache_root": (
                    "path/to/your/refined_namlab"
                ),

                # Directory naming template: auto-generate subdirectory
                # based on parameters, e.g., "L60_pxl_4700".
                # NOTE: the non-purified version has a special naming
                # template below! ❗❗
                "cache_template": "L{level}_{strategy}_{threshold}",

            }
        },
        "depth": {},
    },
    # ==========================
    "datasets": {
        "coco": {
            "root_dir": "path/to/your/MSCOCO",
            "segment_root": "SegmentationClass",
            "categories": [
                'background',
                'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush'
            ]
        },
        "PascalVOC": {
            "root_dir": (
                "path/to/your/VOC2012"
            ),
            # Spliced from root_dir, same below
            "segment_root": "SegmentationClassAug",
            "annotation_toor": "Annotations",
            "train_aug_txt": (
                "ImageSets/SegmentationAug/standard_train_aug.txt"
            ),
            "val_txt": "ImageSets/SegmentationAug/standard_val.txt",
        },
    },
}

# ❗❗ When purification is disabled, override the cache template.
nam_cfg = base_config.get("priors", {}).get("namlab", {})
if nam_cfg and nam_cfg.get("purification_enabled", True) is False:
    nam_cfg["paths"]["cache_template"] = "L{level}_purification_disabled"

config_override = {
    "gpu_ids": "4",         # ⭐⭐
    "rand_seed": 1337,
    "determinism": False,   # Whether to use deterministic operators.
                            # Default: True
    "model_img_size": 1024,
    "box_threshold": 0.3,
    "text_threshold": 0.25,
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 8,
    "num_epochs": 50,
    "max_nums": 50,
    "num_points": 5,
    "resume": False,        # Whether to use a checkpoint ⭐⭐
    # If using a checkpoint, whether to inherit optimizer and scheduler ⭐⭐
    "resume_opt": False,
    "dataset": "COCO",
    "load_type": "soft",
    "prompt": "box"
}

# === Dynamic Prior Request Engine === ⭐⭐
# Core logic: modules declare preprocessing requirements, deduplicated via
# Set. Parameters use a tuple-of-tuples format to guarantee hashability.
base_config['priors']['namlab']['processed_requests'] = set()
base_config['priors']['depth']['processed_requests'] = set()


# ---------------------------------------------------------
# 1. [Loss module: top2_edge_alignment] Requirement registration ⭐⭐
# ---------------------------------------------------------
# Check whether any sub-item weight in the top2_edge_alignment dictionary
# is greater than 0
top2_weights = (base_config['loss']['weights']
                .get('top2_edge_alignment', {}))
if (any(v > 0 for v in top2_weights.values())
        or "top2_edge_alignment_monitoring"
        in base_config["visual"]["probes"]):
    base_config['priors']['namlab']['enabled'] = True
    base_config['priors']['depth']['enabled'] = True

    # Define the exact processing parameter scheme required by this loss
    # function (tuple format, guaranteeing hashable deduplication)
    namlab_req = (
        ('gaussian_sigma', 3.5),
    )
    depth_req = (
        ('apply_norm', True),
        ('gradient_method', 'sobel'),
        # Hill equation: (center_x, center_y, center_slope_k)
        # Example: (0.4, 0.6, 2.5) shifts the energy center toward the
        # upper-left and enhances S-curve contrast
        ('hill_shift', (0.2, 0.7, 1.0)),
        # # Hard threshold: filter out weak gradients below 0.2
        # ('binarize_threshold', 0.2),
        ('gaussian_sigma', 3.5),
    )

    # Inject into the global processing pipeline, informing Model to produce
    base_config['priors']['namlab']['processed_requests'].add(namlab_req)
    base_config['priors']['depth']['processed_requests'].add(depth_req)

    # Establish an entitlement index in prior_configs for losses.py to
    # later verify and assemble key names
    (base_config['model']
     .setdefault('prior_configs', {})
     .setdefault('loss', {})
     .setdefault('top2_edge_alignment', {})
     .update({'namlab': namlab_req, 'depth': depth_req}))

    print(
        "[Config] Logic Trigger: top2_edge_alignment Loss -> "
        "Priors ENABLED."
    )
# ==========================================================


# Generate the final configuration object
cfg = Box(base_config)
cfg.merge_update(config_override)