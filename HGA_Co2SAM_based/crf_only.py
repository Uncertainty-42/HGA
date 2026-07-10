# crf_only.py
"""
Standalone CRF post-processing and evaluation script (Pascal VOC).
Usage:
  python crf_only.py \
      --logits_dir /path/to/logits \
      --output_dir /path/to/preds_crf \
      --results_json /path/to/results_crf.json \
      [--n_jobs 32]
"""
from datetime import datetime
import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./_Optimization_Workspace")))
import time

from utils.dcrf import DenseCRF

class Tee:
    """Simple wrapper that writes simultaneously to a file and the original stream."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def _generate_voc_palette(n=256):
    """
    Generate the standard PASCAL VOC color palette.

    Args:
        n (int): Number of palette entries (default 256).

    Returns:
        np.ndarray: Palette array of shape (256, 3) with RGB values.
    """
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        label = i
        r, g, b = 0, 0, 0
        for j in range(8):
            r |= ((label >> 0) & 1) << (7 - j)
            g |= ((label >> 1) & 1) << (7 - j)
            b |= ((label >> 2) & 1) << (7 - j)
            label >>= 3
        palette[i] = [r, g, b]
    return palette

VOC_PALETTE = _generate_voc_palette(n=256)



def parse_args():
    parser = argparse.ArgumentParser(description="CRF post-processing and evaluation")
    parser.add_argument("--model_path", required=True, type=str, help="Path to model checkpoint (used to locate output root if --output_root not given)")
    parser.add_argument("--infer_set", required=True, type=str, choices=["val","test"], help="Dataset split")
    parser.add_argument("--output_root", type=str, default=None, help="Output root directory (auto-detect if not set)")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--gt_dir", required=True, type=str)
    parser.add_argument("--n_class", type=int, default=21)
    return parser.parse_args()


def _fast_hist(label_true, label_pred, n_class):
    """
    Compute the confusion matrix statistical histogram for a single pair of label maps.

    This method is the lowest-level operator for semantic segmentation evaluation.
    It maps the pairwise relationship between ground-truth and predicted classes to a
    one-dimensional index, leveraging bincount statistics to rapidly construct a
    [n_class, n_class] pixel mapping matrix.

    Args:
        label_true (np.ndarray): Flattened ground-truth label array (1D).
        label_pred (np.ndarray): Flattened prediction array (1D).
        n_class (int): Total number of classes (including background).

    Returns:
        hist (np.ndarray): [n_class, n_class] statistical matrix.
            H[i, j] denotes the number of pixels with true class i predicted as class j.
    """
    mask = (label_true >= 0) & (label_true < n_class)

    if len(label_true) == len(label_pred) and len(label_pred) == len(mask):
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
    else:
        pass
    return hist

def compute_metrics_from_hist(hist, n_class):
    """
    Compute evaluation metrics from a confusion matrix.

    Args:
        hist (np.ndarray): Global confusion matrix of shape [n_class, n_class].
        n_class (int): Total number of classes.

    Returns:
        dict: Dictionary containing Pixel Accuracy, Mean Accuracy,
            Frequency Weighted IoU, Mean IoU, and Class IoU.
    """
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def get_confusion_matrix(label_trues, label_preds, n_class):
    """
    Iterate over sample pairs and accumulate a dataset-level confusion matrix.

    Args:
        label_trues (list[torch.Tensor]): List of ground-truth label tensors.
        label_preds (list[np.ndarray]): List of prediction arrays.
        n_class (int): Total number of classes.

    Returns:
        hist (np.ndarray): Accumulated confusion matrix of shape [n_class, n_class].
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        lt = lt.cpu().numpy()
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist

def scores(label_trues, label_preds, n_class, fabric):
    """
    Compute full semantic segmentation evaluation metrics from sample sequences.

    This function accumulates confusion matrices across all sample pairs to build
    a pixel-distribution view of the entire dataset, then computes Pixel Accuracy (Acc),
    Mean Accuracy (mAcc), and Mean IoU (mIoU) from the global confusion matrix.

    Args:
        label_trues (list[torch.Tensor]): List of ground-truth labels for all samples.
        label_preds (list[np.ndarray]): List of predictions for all samples.
        n_class (int): Total number of classes (including background).
        fabric (L.Fabric, optional): Fabric handle for multi-GPU confusion matrix sync.

    Returns:
        dict: Dictionary containing:
            - "Pixel Accuracy": Global pixel accuracy.
            - "Mean Accuracy": Class-averaged accuracy.
            - "Frequency Weighted IoU": Frequency-weighted IoU.
            - "Mean IoU": Global Mean IoU (mIoU).
            - "Class IoU": Dictionary with per-class IoU values.
    """
    hist = get_confusion_matrix(label_trues, label_preds, n_class)
    if fabric is not None and fabric.world_size > 1:
        hist_tensor = torch.from_numpy(hist).to(fabric.device)
        hist = fabric.all_reduce(hist_tensor, reduce_op="sum").cpu().numpy()
    return compute_metrics_from_hist(hist, n_class)
    
def process_logits_batch(logit_files, output_dir, images_dir, gt_dir, n_class, n_jobs, crf_config, infer_set):
    """
    Process a batch of logits files, generate CRF masks, and save them.
    Skip already existing outputs; do NOT delete logits.
    """
    
    # Worker function (executes in subprocess; loky uses fork since no CUDA context)
    def _job(logit_path, img_path, gt_path, save_path):    
        logit = np.load(logit_path)
        prob = F.softmax(torch.from_numpy(logit), dim=0).numpy()
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        post_processor = DenseCRF(**crf_config)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0).astype(np.uint8)

        # Save color-indexed mask
        pil_img = Image.fromarray(pred, mode='P')
        pil_img.putpalette(VOC_PALETTE.flatten().tolist())
        pil_img.save(save_path)

        # Load GT for scoring
        if gt_path is not None:
            gt = torch.Tensor(np.array(Image.open(gt_path), dtype=np.uint8))
        else:
            gt = torch.zeros(image.shape[:2], dtype=torch.uint8)
        return pred, gt
    
    tasks = []
    for logit_path in logit_files:
        name = logit_path.stem
        img_path = images_dir / f"{name}.jpg"
        save_path = output_dir / f"{name}.png"

        if not img_path.exists():
            print(f"Warning: Image {img_path} not found, skipping {name}")
            continue
        if save_path.exists():
            logit_path.unlink()   # Clean up stale logit early
            continue  # Resume from checkpoint
        if infer_set == "val":
            gt_name = f"{name[-12:] if n_class == 81 else name}.png"
            gt_path = gt_dir / gt_name
            if not gt_path.exists():
                print(f"Warning: GT {gt_path} not found, skipping {name}")
                continue
            tasks.append((logit_path, img_path, gt_path, save_path))
        else:  # Test mode, no GT needed
            tasks.append((logit_path, img_path, None, save_path))

    if tasks:
        if infer_set == "val":
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_job)(t[0], t[1], t[2], t[3]) for t in tasks
            )
        else:
            # Test set: pass gt_path as None; _job must handle it internally
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_job)(t[0], t[1], None, t[3]) for t in tasks
            )
        return results
    return []

def main(args):
    # Determine output root directory
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        # Auto-detect the most recent output directory under the model directory
        ckpt_dir = Path(args.model_path).parent
        pattern = f"{Path(args.model_path).stem}_*_{args.infer_set}"
        time.sleep(60)
        candidates = sorted(ckpt_dir.glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No output directory found matching {pattern} in {ckpt_dir}")
        output_root = candidates[0]
        print(f"Auto-detected output root: {output_root}")
        
    # Logging setup
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    crf_log = open(logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_crf.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, crf_log)
    sys.stderr = Tee(sys.stderr, crf_log)

    logits_dir = output_root / "logits"
    if args.infer_set == "val":
        output_dir = output_root / "preds_crf"
    else:
        output_dir = output_root / "crf" / "results" / "VOC2012" / "Segmentation" / "comp6_test_cls"

    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _ = torch.zeros(1).softmax(dim=0)   # Force-trigger low-level init to ensure single-threaded execution
    n_jobs = args.n_jobs or int(os.cpu_count() * 0.4)
    print(f"CRF parallel jobs: {n_jobs}")

    # CRF parameters (original DenseCRF configuration)
    crf_config = dict(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Total number of samples (based on image count)
    total_samples = len(list(images_dir.glob("*.jpg"))) if args.n_class == 81 else 1449 if args.infer_set == "val" else 1456
    print(f"Total samples expected: {total_samples}")

    # ===== CRF Dispatch Loop =====
    while True:
        logit_files = sorted(logits_dir.glob("*.npy"))
        if logit_files:
            print(f"Found {len(logit_files)} logits, starting processing...")
            process_logits_batch(
                logit_files, output_dir, images_dir, gt_dir,
                args.n_class, n_jobs, crf_config, args.infer_set
            )
            # Delete processed logits
            for lf in logit_files:
                try:
                    lf.unlink()
                except FileNotFoundError:
                    pass
            print(f"Processed and cleaned up {len(logit_files)} logits.")
            continue  # Re-scan immediately
        else:
            print("logits empty, waiting 30s to confirm...")
            time.sleep(30)
            logit_files = sorted(logits_dir.glob("*.npy"))
            if logit_files:
                print("New logits detected, waiting 10 min before continuing...")
                time.sleep(600)
                continue
            else:
                print("No new logits, checking completeness...")
                pred_files = list(output_dir.glob("*.png"))
                pred_count = len(pred_files)
                if pred_count == total_samples:
                    print("All samples processed.")
                    break
                else:
                    print(f"Warning: Only {pred_count}/{total_samples} samples processed so far; inference may be incomplete or some samples are missing.")
                    continue
                    all_names = {p.stem for p in images_dir.glob("*.jpg")}
                    pred_names = {p.stem for p in output_dir.glob("*.png")}
                    missing = all_names - pred_names
                    if missing:
                        print("Missing samples:")
                        for m in sorted(missing):
                            print(f"  - {m}")
                    print("Exiting.")
                    return

    # ===== Final Scoring (validation set only) =====
    if args.infer_set == "val":
        print("Computing CRF post-processing scores...")
        # preds = []
        # gts = []
        hist = np.zeros((args.n_class, args.n_class), dtype=np.int64)
        valid_count = 0
        for p in tqdm(sorted(output_dir.glob("*.png")), desc="CRF scoring"):
            name = p.stem
            gt_name = f"{name[-12:] if args.n_class == 81 else name}.png"
            gt_path = gt_dir / gt_name
            if gt_path.exists():
                pred = np.array(Image.open(p), dtype=np.uint8)
                gt_arr = np.array(Image.open(gt_path), dtype=np.uint8)
                hist += _fast_hist(gt_arr.flatten(), pred.flatten(), args.n_class)
                valid_count += 1
            else:
                print(f"Warning: GT {gt_path} missing, skipping {name}")
        if valid_count > 0:
            score = compute_metrics_from_hist(hist, args.n_class)
            results_json = output_root / "results_crf.json"
            results_json.write_text(json.dumps(score, indent=4, sort_keys=True))
            print(f"CRF Mean IoU: {score['Mean IoU']:.4f}")
            print(f"Results saved to {results_json}")
        else:
            print("No valid samples for scoring.")
    else:
        print("Test set CRF post-processing complete.")


if __name__ == "__main__":
    
    args = parse_args()

    model_path = Path(args.model_path)
    ckpt_dir = model_path.parent  # .../output/.../ckpt
    exp_dir = ckpt_dir.parent     # .../output/...
    main(args)