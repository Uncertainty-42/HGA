This directory contains the fully refactored and upgraded **Co2SAM-based training framework**, serving as the primary validation host for the **Hierarchical-Geometric Alignment (HGA)** paradigm. It supports the original ViT-B SAM and lightweight EfficientViT-SAM-XL0 backbones, standard CT (transposed convolution) and patched RC (Resize-Convolution) decoders, and online HGA constraints. This workspace is optimized for reproducing the peak benchmarks reported in the paper on both the PASCAL VOC 2012 and MS COCO 2014 datasets.

---

## 📂 Expected Directory Structure

Before setting up your environment, ensure your workspace matches the following layout:

```text
HGA_Co2SAM_based/
├── train_voc.py / train_coco.py      # Training entries
├── val_voc.py / val_coco.py          # Validation scripts
├── crf_only.py                       # Offline CRF evaluation script
├── requirements.txt                  # Python dependencies
├── SPECIAL_INSTALL.md                # Specialized compilation guide
├── checkpoints/                      # Directory for pretrained weights
├── datasets/                         # Data loader implementations
├── utils/                            # Evaluation utilities
└── _Optimization_Workspace/          # Turnkey experiment sandboxes
    ├── Exp000_final_results_for_VOC/ # Read-only VOC results showroom
    ├── Exp000_final_results_for_COCO/# Read-only COCO results showroom
    ├── Exp001_reproduce_here/        # ★ Turnkey sandbox for your runs
    ├── templates/                    # Global config templates
    ├── modules/                      # HGA loss and RC decoder components
    └── tools/                        # Auxiliary workspace utilities
```

---

## ⚙️ 1. Environment Setup

Executing these large visual foundation models requires compiling specific CUDA extensions. Follow this step-by-step sequence to prepare an isolated Conda environment:

```bash
# Create and activate an isolated environment
conda create -n hga_env python=3.8.20 -y
conda activate hga_env

# Install PyTorch and compatible CUDA dependencies
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install standard dependencies
pip install -r requirements.txt

# Compile and install detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install -e . && cd ..

# Compile and install GroundingDINO (Requires CUDA_HOME set)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

# Install Segment Anything (SAM) and EfficientViT submodules
pip install git+https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit && pip install -e . && cd ..

# [Optional] Compile pydensecrf for offline post-processing evaluation
# Note: You can safely skip this in 'hga_env' as we highly recommend using 
# the dedicated 'hga_crf_env' below for much faster CRF processing.
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

> 💡 **Troubleshooting & Checkpoint Downloads**: If you encounter any compilation errors (especially during GroundingDINO or pydensecrf compilation), please read [**`SPECIAL_INSTALL.md`**](SPECIAL_INSTALL.md) for detailed, step-by-step resolution steps.

---

## 💾 2. Pretrained Weight Acquisition

Download the following three official pretrained checkpoints and place them directly under the `checkpoints/` directory:

| Model Component | Filename | Download Source |
|:---|:---|:---:|
| **Segment Anything (SAM-ViT-B)** | `sam_vit_b_01ec2d.pth` | [SAM Official](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec2d.pth) |
| **Grounding DINO (Swin-T OGC)** | `groundingdino_swint_ogc.pth` | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| **EfficientViT-SAM (XL0)** | `efficientvit_sam_xl0.pt` | [EfficientViT](https://github.com/mit-han-lab/efficientvit) |

---

## 📊 3. Dataset & Priors Organization

HGA relies on both 2D hierarchical region contours (NAMLab) and 3D relative depth gradients. Set up your directories exactly as shown below:

```text
### 1. PASCAL VOC 2012 Layout
├── ImageSets/
│   └── SegmentationAug/       # Dataset partition files (train.txt, val.txt)
├── JPEGImages/                # Original source images (.jpg)
├── SegmentationClassAug/      # Ground Truth annotation masks (.png)
└── priors/                    # Sibling directory for structural priors
    ├── depth_npy/             # Relative depth maps (.npy files)
    ├── namlab_pt/             # Pre-computed raw NAMLab hierarchical priors (.pt files)
    └── refined_namlab/        # [Automatically Created] Purified caching directory
        └── L60_pxl_4700/      # [Automatically Created] Purified hierarchical priors

### 2. MS COCO 2014 Layout
├── JPEGImages/                # Original source images partitioned into splits
│   ├── train/                 
│   └── val/                   
├── SegmentationClass/         # Ground Truth annotation masks partitioned into splits
│   ├── train/                 
│   └── val/                   
└── priors/                    # Sibling directory for structural priors
    ├── depth_npy/
    │   ├── train/
    │   └── val/
    ├── namlab_pt/       # Pre-computed raw COCO NAMLab priors (.pt files)
    │   ├── train/
    │   └── val/
    └── refined_namlab/        # [Automatically Created] Purified caching directory
        └── L60_pxl_4700/
            ├── train/
            └── val/
```

---

## ⚙️ 4. NAMLab Purification On-The-Fly Caching Mechanism

To optimize local storage and eliminate the overhead of manually pre-generating massive processed files, our framework implements an elegant, on-the-fly caching mechanism for hierarchical priors:

1. At the initialization of the training loop, the data loader automatically checks if the targeted purification folder `priors/refined_namlab/L{level}_{strategy}_{threshold}/` exists (`L60_pxl_4700` for example).
2. **If missing**: The loader reads raw hierarchical prior arrays from `namlab_pt/`, executes the hierarchical region purification algorithm in memory (using the layer 60 structure and a purity threshold of 4700), automatically creates the sub-directory on disk, caches the generated `.pt` results there, and utilizes them for training.
3. **If present**: The loader directly reads the purified priors from `L60_pxl_4700/`, bypassing redundant computations.

You only need to download the raw NAMLab priors from Hugging Face Datasets and place them in the corresponding `namlab_pt/` directories.

## 💾 5. Prior Generation & Downloading:
*   **NAMLab Priors**: Download pre-computed `.pt` files directly from Hugging Face Datasets:
    *   [PASCAL VOC 2012 NAMLab Priors](https://huggingface.co/datasets/Uncertainty-42/PascalVOC_NAMLab_pt)
    *   [MS COCO 2014 NAMLab Priors](https://huggingface.co/datasets/Uncertainty-42/MSCOCO_NAMLab_pt)
*   **Depth Priors**: Generate them locally to save storage space. Run `generate_depth_prior.py` (located at the repository root), editing `SOURCE_DIR` and `OUTPUT_DIR` inside the script to point to your respective image directories.


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

## 🚀 6. How to Run: Turnkey Reproduction Workflow

To minimize path-routing errors and avoid modifying globally shared files, we have prepared a turnkey sandboxed workspace at `_Optimization_Workspace/Exp001_reproduce_here/`. 

```bash
# Step 1: Navigate to the pre-configured sandbox
cd _Optimization_Workspace/Exp001_reproduce_here/
```

### Step 2: Choose Your Target Configuration
Open the read-only results showroom under `Exp000_final_results_for_VOC/` or `Exp000_final_results_for_COCO/`. Read the `parameters_summary.md` file inside to find the exact parameter dictionary for your target run (e.g., `Eff-XL0+RC+HGA`).

### Step 3: Configure `config.py`
Open `_Optimization_Workspace/Exp001_reproduce_here/config.py`. Update the following:
*   Set local dataset and prior paths (e.g., `data_root`, `namlab_pt_root`, etc.) to match your directory structures.
*   Copy and paste the target parameters from the `parameters_summary.md` into your `config.py`.

### Step 4: Launch Training
Execute the pre-configured shell script:
```bash
bash run.sh
```
The script will run the training entry point dynamically relative to your directory, keeping the root directory clean of intermediate outputs.

---

## 📊 7. Evaluation, Logging, & Auditing

Every training run automatically generates a unique, timestamped output subdirectory containing:
*   `ckpt/`: Model checkpoints, along with training-stage evaluation scores.
*   `log/`: Standard `console.log` files and manual backups of your `config.py`.
*   `viz/`: Qualitative image plots (original image, DINO boxes, mask overlays, heatmaps, and false-color composites).

### Verification & Reproducibility Auditing:
To verify your results against the paper's benchmark runs, navigate to the relevant showroom subdirectory in `Exp000_final_results_for_VOC/` or `Exp000_final_results_for_COCO/` (e.g., `Eff-XL0+RC+HGA/ckpt/u_teacher_model_..._val/`).

You will find three target JSON score files generated during independent validation:
1.  `log_epochX_originalY.json`: Training-stage validation metrics.
2.  `result.json`: Comprehensive class-by-class validation metrics **before applying CRF** post-processing.
3.  `result_crf.json`: Comprehensive class-by-class validation metrics **after applying CRF** post-processing.

> 💡 **Note on MS COCO Evaluation Discrepancy**: During MS COCO training, validation is performed on a uniformly downsampled subset (1/20 of the validation set) to save training time, saving metrics with `fast` in their filenames (e.g., `..._fast.json`) to monitor trends. In contrast, `val_coco.sh` evaluates the complete, unsampled validation set. Consequently, the final scores (`result.json`) will exhibit slight, expected discrepancies compared to these intermediate `fast` training metrics. PASCAL VOC always uses full evaluation.

To evaluate your trained checkpoints, we provide consolidated evaluation scripts **`val_voc.sh`** and **`val_coco.sh`** at the repository root.

> ⚠️ **Mandatory Prerequisite**: The evaluation scripts dynamically reconstruct your model based on the configuration used during training. Therefore, before executing evaluation, ensure that the target checkpoint's timestamped output directory contains the archived `config.py` file within its `log/` folder (e.g., `[your_timestamp_dir]/log/config.py`).

### ⚙️ DenseCRF Optimization & Environment Setup (`hga_crf_env`)
While both training and inference are performed inside your primary environment (`hga_env`), executing CPU-bound DenseCRF post-processing under PyTorch 2.x and CUDA 12.1 faces severe execution bottlenecks. 

To resolve this, we leverage a dedicated, lightweight virtual environment named **`hga_crf_env`** specifically for the CRF processing stage. The evaluation shell scripts **automatically handle the environment routing and switching under the hood**; you only need to pre-configure the `hga_crf_env` environment once:
> 💡 **Note:** If you set up `hga_crf_env` as described below, installing `pydensecrf` in your primary `hga_env` is **not required**.
```bash
# Create the dedicated CRF environment
conda create -n hga_crf_env python=3.7.12 -y
conda activate hga_crf_env

# Install lightweight dependencies
pip install torch==1.12.1 numpy==1.21.5 pillow==9.4.0 joblib==1.3.2 tqdm==4.65.0

# Compile pydensecrf locally
git clone https://github.com/lucasb-eyer/pydensecrf.git
cd pydensecrf && pip install . && cd ..
```
### Running Evaluation:
To execute evaluation, you only need to configure the parameters directly inside the shell scripts and run them. 

1. Open the target evaluation script (**`val_voc.sh`** or **`val_coco.sh`**) located at the repository root.
2. Edit the following parameters inside the script:
   *   `MODEL_PATH`: Set this to the absolute path of your target checkpoint `.pth` file (ensure its archived `log/config.py` is present in the checkpoint's parent folder).
   *   `OUTPUT_ROOT`: 
       *   Leave blank (`""`) to automatically generate a new, timestamped output directory for a fresh evaluation run.
       *   Specify an existing output directory path to resume an interrupted evaluation from where it left off (breakpoint continuation).
   *   `IMAGES_DIR` & `GT_DIR` (located inside the CRF block): Configure these paths to point to your local dataset directories.
3. Activate your primary environment and run the script directly:

```bash
# 1. Ensure you are in your primary environment
conda activate hga_env

# 2. Run the evaluation script
# For PASCAL VOC 2012:
bash val_voc.sh

# For MS COCO 2014:
bash val_coco.sh
```
---

## 🙏 Acknowledgements

This workspace is built upon code and architectural designs from the following open-source frameworks. We express our gratitude to their respective authors:

*   **Co2SAM**: [Co2SAM Repository](https://github.com/Yunxiao-Liu/Co2SAM)
*   **Segment Anything (SAM)**: [segment-anything](https://github.com/facebookresearch/segment-anything)
*   **Grounding DINO**: [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
*   **Detectron2**: [detectron2](https://github.com/facebookresearch/detectron2)
*   **Depth Anything V2**: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
*   **NAMLab Framework**: [NAMLab](https://github.com/YunpingZheng/NAMLab)
*   **DenseCRF / pydensecrf**: We sincerely thank the authors of the fully connected Conditional Random Field (DenseCRF) framework and its Python wrapper `pydensecrf` [lucasb-eyer/pydensecrf](https://github.com/lucasb-eyer/pydensecrf) for providing the post-processing baseline code used in our comparative evaluations.