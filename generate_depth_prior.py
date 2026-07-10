"""
Offline Depth Prior Generation Utility for HGA.

This script is a batch-processing wrapper designed to recursively scan a dataset 
directory (e.g., PASCAL VOC or MS COCO), infer relative depth maps, and save 
them as lightweight NumPy arrays (.npy) while preserving the original directory structure.

Third-Party Attribution:
------------------------
This script utilizes the pre-trained 'Depth Anything V2' model hosted on Hugging Face.
All intellectual property rights and core implementations regarding the depth estimation 
model belong to the original authors.
- Paper: Depth Anything V2 (NeurIPS 2024)
- Original Repository: https://github.com/DepthAnything/Depth-Anything-V2
- HF Model Page: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

Dependencies:
-------------
    This script requires the official Depth-Anything-V2 package and its weights.
    To run this utility, ensure you have installed:
        pip install torch torchvision opencv-python pillow
    And follow instructions at the original respotory.

Usage:
------
Modify the 'SOURCE_DIR' and 'OUTPUT_DIR' variables in the '__main__' block to point 
to your local image directory and desired prior output directory, respectively.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def generate_depth_maps(source_root: Path, target_root: Path, model, processor, device):
    """
    Recursively scan source_root for image files, generate depth maps,
    and save them as .npy files under target_root preserving the directory structure.
    """
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Collect all image file paths recursively
    image_paths = []
    for root, _, files in os.walk(source_root):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(Path(root) / file)

    if not image_paths:
        print(f"No image files found under {source_root}")
        return

    print(f"Found {len(image_paths)} image files. Starting depth map generation...")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Generating depth maps"):
            # Compute relative path from source_root
            rel_path = img_path.relative_to(source_root)
            # Corresponding .npy path under target_root
            npy_path = target_root / rel_path.with_suffix(".npy")

            # Skip if already exists
            if npy_path.exists():
                continue

            try:
                # Load image
                pil_image = Image.open(img_path).convert("RGB")

                # Preprocess
                inputs = processor(images=pil_image, return_tensors="pt").to(device)

                # Inference
                outputs = model(**inputs)

                # Post-process to original size
                prediction = processor.post_process_depth_estimation(
                    outputs,
                    target_sizes=[(pil_image.height, pil_image.width)]
                )
                depth_np = prediction[0]["predicted_depth"].cpu().numpy()  # (H, W)

                # Create parent directory if needed
                npy_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as .npy
                np.save(npy_path, depth_np)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print("All done.")


if __name__ == "__main__":
    # ======== User-configurable parameters ========
    # The script recursively walks through SOURCE_DIR, preserving the full
    # subdirectory hierarchy. For each image found (e.g., train/xxx.jpg or
    # val/yyy.png), the corresponding depth map is saved under OUTPUT_DIR
    # with the same relative path (e.g., OUTPUT_DIR/train/xxx.npy).
    # Existing .npy files are skipped to allow resuming.
    SOURCE_DIR = Path("path/to/your/orig_dir")   # e.g., COCO/JPEGImages or VOC/JPEGImages
    OUTPUT_DIR = Path("path/to/your/output_dir") # where depth maps will be saved

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # Load model and processor
    print(f"Loading model '{MODEL_ID}'...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE)
    print("Model loaded.")

    # Generate depth maps
    generate_depth_maps(SOURCE_DIR, OUTPUT_DIR, model, processor, DEVICE)