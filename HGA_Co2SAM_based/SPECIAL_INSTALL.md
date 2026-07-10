# Special Dependencies Installation

## Installation

### Segment-Anything
```bash
pip install -e git+https://github.com/facebookresearch/segment-anything.git#egg=segment_anything
```

### GroundingDINO
Clone the repository and create a symlink to it inside your project root:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
ln -s /path/to/GroundingDINO ./GroundingDINO
```

> The symlink name must be `GroundingDINO` and must be placed at the root of `HGA_Co2SAM_based`.

### EfficientViT
```bash
pip install -e git+https://github.com/mit-han-lab/efficientvit.git#egg=efficientvit
```

### pydensecrf
```bash
git clone https://github.com/lucasb-eyer/pydensecrf.git
cd pydensecrf
pip install .
```

---

## Pretrained Weights

Download the following files into `./checkpoints/` (or any directory you prefer; update `config.py` accordingly):

- SAM ViT-B:  
  `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

- GroundingDINO Swin-T:  
  `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`

- EfficientViT-SAM XL0:  
  `https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt`

---

## Verification

Run the following to confirm all special dependencies are correctly installed:

```bash
python -c "import segment_anything; import groundingdino; import efficientvit; import pydensecrf; print('OK')"
```

If you encounter any build errors with `pydensecrf`, ensure your system has a C++ compiler and Python development headers.
