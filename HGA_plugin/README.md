# HGA: Standalone Plug-and-Play Toolkit

This directory provides the **standalone, lightweight plugin version** of the Hierarchical-Geometric Alignment (HGA) paradigm. 

It is designed as an architecture-agnostic boundary internalization engine. You can seamlessly mount this toolkit onto any existing Weakly Supervised Semantic Segmentation (WSSS) baseline (e.g., traditional CAM-based networks, VLM-alignment paradigms, etc.) to achieve native, sharp boundary predictions without relying on offline post-processing.

**Core Philosophy: Zero Intrusion.**  
Integrating HGA requires **zero modifications** to your existing backbone, mask decoder, or base training objectives. You simply intercept the output multi-class logits, process the structural priors, and superimpose the returned HGA alignment constraints onto your existing base loss.

---

## 📂 Toolkit Directory Structure

To use this plugin, simply copy the entire `priors_optimization/` directory into your project's codebase.

```text
priors_optimization/
├── losses.py               # Core boundary alignment & thinning constraints ('HGACriterion')
├── preprocess.py           # Engine to synthesize continuous Attractor Potential Fields
└── namlab_refinement/      # Sub-package for offline hierarchical region restoration & purification
    ├── __init__.py         # Exposes 'run_namlab_refinement'
    ├── core.py             
    ├── merger.py           
    ├── purifier.py         
    └── utils.py            
```

---

## ⚙️ Environment Requirements

This toolkit is designed to be extremely lightweight and requires only fundamental scientific computing libraries:
*   `torch` (PyTorch for tensor operations and loss gradients)
*   `numpy`
*   `scikit-image` (Required strictly by `namlab_refinement/utils.py` for standard D50 RGB-to-Lab color space conversions)

---

## 🚀 Integration in 3 Steps

Integrating HGA into your custom WSSS training loop involves three precise steps: Data Preparation, Criterion Initialization, and Forward Constraint.

### Step 1: Data Preparation & The Spatial Sync Contract
Inside your Dataset's `__getitem__` method, you must load the raw priors, optionally refine the NAMLab fragments (usually cached after the first epoch), and pass them to the `Preprocessor`.

> ⚠️ **CRITICAL: The Spatial Synchronization Contract**
> For HGA's geometric guidance to function correctly, the prior tensors and your main RGB image **must undergo the exact same spatial augmentations** (cropping, scaling, flipping).
> *   **Discrete Maps** (NAMLab index maps and Validity Masks) **must** use Nearest Neighbor interpolation.
> *   **Continuous Maps** (Depth maps) **must** use Bilinear or Bicubic interpolation.

```python
# Example inside your Dataset's __getitem__
from priors_optimization.namlab_refinement import run_namlab_refinement
from priors_optimization.preprocess import Preprocessor

# 1. Load your raw priors
# Load the raw NAMLab .pt file downloaded from Hugging Face
pt_data = torch.load("path/to/your/namlab_file.pt") 
depth_raw = np.load("path/to/your/depth_file.npy")

# 2. Refine NAMLab regions (Typically, you would cache this result to disk after the 1st run)
refined_namlab_map = run_namlab_refinement(
    pt_data=pt_data, 
    img_rgb=img_rgb, 
    depth_raw=depth_raw,
    target_level=60,
    purification_enabled=True,
    area_threshold=4700.0
)

# 3. APPLY SPATIAL AUGMENTATIONS HERE
# (Ensure refined_namlab_map, depth_raw, and the image are cropped/flipped/scaled identically)

# 4. Process into continuous Attractor Potential Fields
namlab_blurred, depth_blurred = Preprocessor.prepare_processed_priors(
    namlab_raw=refined_namlab_map, 
    depth_raw=depth_raw, 
    sigma=3.5, 
    hill_params=(0.2, 0.7, 1.0) # Slope k=1.0 or 2.0 based on your preference
)

# 5. Create a Geometric Validity Mask (Optional but recommended for padded images)
# 1.0 for valid image areas, 0.0 for padding areas
valid_mask = generate_valid_mask(...) 

return image, label, namlab_blurred, depth_blurred, valid_mask
```

### Step 2: Initialize the HGACriterion
In your main training script, instantiate the arbiter alongside your base losses. Adjust the `num_classes` to match your dataset.

```python
from priors_optimization.losses import HGACriterion

# Initialize the HGA Arbiter
hga_criterion = HGACriterion(
    use_priors=True,
    w_align_nam=0.2,   # Scale these weights to match the magnitude of your base loss
    w_align_dep=0.2,
    w_thin=0.2,
    threshold=10.0,
    num_classes=21     # e.g., 21 for Pascal VOC, 81 for MS COCO
).cuda()
```

### Step 3: Forward Pass & Loss Accumulation
Intercept the raw logits from your network, compute the HGA losses, and accumulate them.

```python
for images, labels, nam_blurred, dep_blurred, valid_mask in dataloader:
    
    # 1. Forward pass through your baseline network
    # seg_logits shape: [B, num_classes, H, W]
    seg_logits = model(images)
    base_loss = compute_your_base_loss(seg_logits, labels)
    
    # 2. Compute HGA boundary internalization constraints
    # Ensure seg_logits and prior fields have the same spatial resolution (H, W)
    hga_loss_dict = hga_criterion(seg_logits, nam_blurred, dep_blurred, valid_mask)
    
    # 3. Superimpose and backpropagate
    total_loss = base_loss + sum(hga_loss_dict.values())
    total_loss.backward()
```

---

## 💾 Core Prior Data Acquisition


HGA relies on two offline priors per training image, which are loaded on-the-fly by our DataLoader. You only need to obtain them once before training:

1.  **NAMLab Hierarchical Priors (pre-computed `.pt` files)**:
    The purified region priors have been uploaded to Hugging Face Datasets. Download and place them in your configured directories:
    *   [PASCAL VOC 2012 NAMLab Priors](https://huggingface.co/datasets/Uncertainty-42/PascalVOC_NAMLab_pt)
    *   [MS COCO 2014 NAMLab Priors](https://huggingface.co/datasets/Uncertainty-42/MSCOCO_NAMLab_pt)
2.  **Depth Geometric Priors (locally generated)**:
    To avoid transferring gigabytes of redundant files, we provide a standalone script `generate_depth_prior.py` that utilizes **Depth Anything V2** to extract depth maps locally and save them as lightweight `.npy` arrays.
    > 💡 **Integration Note:** The `generate_depth_prior.py` script is located at the **repository root**. If you are integrating this toolkit as a standalone plugin, please copy that script from the root to your workspace to generate your depth arrays.


### 📥 Downloading & Extraction Guide

We recommend using the official `huggingface-cli` to download the priors. This tool supports robust chunked downloads and automatic resume capabilities. 

> **Tip (For users with restricted internet access):** If you are unable to connect to the Hugging Face global servers directly, you can prepend `HF_ENDPOINT=https://hf-mirror.com` to any download command below to route through the official mirror.

---

#### 1. PASCAL VOC 2012 NAMLab Priors (Single Archive)
The VOC dataset priors are packaged into a single archive (`priors_voc.zip`). 

Navigate to your local `priors/` directory (where `namlab_pt` should be created) and run:

```bash
# Step 1: Navigate to your VOC priors folder
cd /path/to/VOCdevkit/VOC2012/priors

# Step 2: Download the single zip file (approx. 4.2 GB)
huggingface-cli download Uncertainty-42/PascalVOC_NAMLab_pt priors_voc.zip --repo-type dataset --local-dir ./

# Step 3: Extract the contents into namlab_pt/
unzip priors_voc.zip -d namlab_pt/

# Step 4: Clean up the archive (Optional)
rm priors_voc.zip
```

---

#### 2. MS COCO 2014 NAMLab Priors (Split-Volume Archive)
Because the COCO dataset priors (approx. 63.69 GB in total) exceed Hugging Face's 50 GB individual file upload limit, the archive is hosted as a split-volume archive: from `priors_coco.zip.part1` to `priors_coco.zip.part3`. You must concatenate them locally before extraction.

Navigate to your local `priors/` directory and run:

```bash
# Step 1: Navigate to your COCO priors folder
cd /path/to/MSCOCO/priors

# Step 2: Download both split parts (approx. 30 GB, 30 GB and 3.69 GB respectively)
huggingface-cli download Uncertainty-42/MSCOCO_NAMLab_pt priors_coco.zip.part1 priors_coco.zip.part2 priors_coco.zip.part3 --repo-type dataset --local-dir ./

# Step 3: Concatenate the split parts back into a single valid zip archive
cat priors_coco.zip.part* > priors_coco.zip

# Step 4: Extract the contents into namlab_pt/
# (This will automatically recreate the train/ and val/ sub-directories)
unzip priors_coco.zip -d namlab_pt/

# Step 5: Clean up the temporary archive and parts (Optional)
rm priors_coco.zip.part* priors_coco.zip
```

---

## 🙏 Acknowledgements

We express our sincere gratitude to the authors of the following foundational works that inspired and enabled the structural priors utilized in this toolkit:
*   **NAMLab Framework**: [NAMLab](https://github.com/YunpingZheng/NAMLab) (Providing the hierarchical invariant region partition principles).
*   **Depth Anything V2**: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (Providing the highly robust monocular relative depth estimation foundations).

## 📄 License  

This toolkit is released under the **MIT License**. See the `LICENSE` file in the project root for details.

