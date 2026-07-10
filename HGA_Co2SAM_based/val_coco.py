# val_voc.py
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./_Optimization_Workspace")))

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import lightning as L
from model import Model
from groundingdino.util.inference import Model as dino_model
from datasets import call_load_dataset
from utils.eval_utils_coco import sem_seg_postprocess, _draw_validation_visuals, _fast_hist, compute_metrics_from_hist

from _Optimization_Workspace.tools.visualize import VISUALIZER, VOC_PALETTE

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

# Resolve missing training entry class issue (Co2SAMLRSchedule, etc.)
import train_coco
sys.modules['__main__'].Co2SAMLRSchedule = train_coco.Co2SAMLRSchedule

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/ExCEL/00_exp/00_ablation/main_ablation/baseline/checkpoints/model_iter_30000.pth", type=str, help="model_path")
parser.add_argument("--crf_post", action="store_true", help="apply CRF post-processing")
parser.add_argument("--scales", default=[0.7, 1.0, 1.2, 1.5], help="multi_scales for seg")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--gpu", default="0", type=str, help="GPU device id")

parser.add_argument("--output_root", default=None, type=str, help="Output root directory. Auto-create from model path if not set.")

parser.add_argument("--viz_eval_enabled", action="store_true", help="enable visualization in eval mode")
parser.add_argument("--viz_probes", default="eval_visuals", type=str, help="visuals subdirectory name")

parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--background_class", default=0, type=int, help="background class index")

parser.add_argument("--viz_warmup_steps", default=0, type=int, help="visualization policy: warmup steps")
parser.add_argument("--viz_warmup_freq", default=1, type=int, help="visualization policy: warmup frequency")
parser.add_argument("--viz_epoch1_freq", default=1, type=int, help="visualization policy: epoch1 frequency")
parser.add_argument("--viz_later_freq", default=1, type=int, help="visualization policy: later frequency")

def _infer_single(model, dino, cfg, args, image, gt_mask, image_path, class_label, priors_payload):
    """
    Single-sample inference, fully replicating the core logic of eval_utils.validate.
    Returns:
        pseudo_label: numpy uint8, (H_orig, W_orig)
        r_mask: numpy float32, (num_classes, H_orig, W_orig) logits
    """
    dino_class_names = [ 
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                'background',
            ]

    valid_size = image.shape[-2:]
    dino_images = (image.squeeze(0).permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8).copy()
    detections = dino.predict_with_classes(
        dino_images, dino_class_names, cfg.box_threshold, cfg.text_threshold
    )

    if VISUALIZER.is_active("dino_boxes"):
        VISUALIZER.draw_detections(dino_images, detections, dino_class_names, tag="Dino_Boxes", title="Validation Image + DINO Boxes")

    boxes = torch.tensor(detections.xyxy)
    class_ids = detections.class_id
    voc_class_ids = []
    for cid in class_ids:
        if cid == 80:
            voc_class_ids.append(0)
        else:
            voc_class_ids.append(cid + 1)
    voc_class_ids = torch.tensor(voc_class_ids, dtype=torch.long)

    r = torch.zeros((args.num_classes, 1024, 1024))
    height, width = gt_mask[0].shape

    if len(boxes) > 0:
        transformed_boxes = model.apply_boxes_torch1(boxes, dino_images.shape[:2])
        # print("transformed_boxes:", transformed_boxes)
        dino_images_resize = torch.tensor(dino_images).unsqueeze(0).permute(0, 3, 1, 2).cuda().float()
        _, pred_masks, _, _ = model(dino_images_resize, (transformed_boxes.cuda().to(torch.float64),), (1024, 1024))
        masks = pred_masks[0]   # Without DINO confidence weighting

        # Preserve prior processing for future audit visualization
        model.prepare_for_image(priors_payload)

        for unique_cid in voc_class_ids.unique():
            if unique_cid == 0:
                continue
            class_masks = masks[voc_class_ids == unique_cid]
            if len(class_masks) > 0:
                r[unique_cid] = class_masks.max(dim=0)[0]
    
        
        foreground_max = r[1:].max(dim=0).values
        r[0, foreground_max <= 0] = 1.0

        # --- 1024-space visualization audit ---
        _draw_validation_visuals(
            r_1024=r.clone(),
            image_path=image_path,
            gt_mask_orig=gt_mask,
            visualizer=VISUALIZER,
            priors_payload=priors_payload,
            model_instance=model,
            raw_image_1024=image,
            cfg=cfg
        )
        r_mask = sem_seg_postprocess(r, valid_size, height, width).cpu().numpy()

        if cfg.get("eval_with_oracle_filter", False):
            new_r_mask = np.zeros_like(r_mask)
            new_r_mask[0] = r_mask[0]
            active_classes = torch.nonzero(class_label[0]).squeeze(1)
            for cls_idx in active_classes:
                idx = cls_idx.item()
                if 0 < idx < args.num_classes:
                    new_r_mask[idx] = r_mask[idx]
            pseudo_label = np.argmax(new_r_mask, axis=0).astype(np.uint8)
        else:
            pseudo_label = np.argmax(r_mask, axis=0).astype(np.uint8)
    else:
        r_mask = np.zeros((args.num_classes, height, width), dtype=np.float32)
        r_mask[0, :, :] = 1.0
        pseudo_label = np.zeros((height, width), dtype=np.uint8)
        
    unique_final = np.unique(pseudo_label)
    # print(f"[DIAG] Final pseudo_label unique classes = {unique_final.tolist()}")
    return pseudo_label, r_mask


if __name__ == "__main__":

    args = parser.parse_args()

    # 1. Auto-locate cfg
    model_path = Path(args.model_path)
    ckpt_dir = model_path.parent  # .../output/.../ckpt
    exp_dir = ckpt_dir.parent     # .../output/...
    cfg_path = exp_dir / "log" / "config.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration backup file not found: {cfg_path}")
    spec = importlib.util.spec_from_file_location("exp_config", str(cfg_path))
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.cfg

    # 2. Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 3. Environment precision and determinism alignment (must be strictly consistent with training)
    torch.set_float32_matmul_precision('high') # type: ignore
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore
    torch.backends.cudnn.allow_tf32 = True # type: ignore
    if cfg.rand_seed is not None:
        train_coco.enforce_strict_determinism(cfg.rand_seed)

    # 3. Lightweight Fabric (to satisfy load_datasets' requirement for global_rank)
    fabric = L.Fabric(devices=1, accelerator="auto")
    fabric.launch()
    if cfg.rand_seed is not None:
        fabric.seed_everything(cfg.rand_seed)

    # 4. Build validation DataLoader (reuse training logic)
    load_datasets = call_load_dataset(cfg)
    _, val_data = load_datasets(cfg, cfg.model_img_size or 1024, fabric)
    val_data = fabric.setup_dataloaders(val_data)
    print(f"Validation set sample count: {len(val_data.dataset)}")

    # 5. Initialize model and load weights
    model = Model(cfg)
    model.setup()
    model.apply_structure_patches()
    state_dict = torch.load(args.model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"⚠️ Missing structures: {missing}")
    print(f"⚠️ Redundant structures: {unexpected}")
    model.eval().cuda()

    # 6. Initialize DINO
    cache_config_file = "path/to/your/GroundingDINO/GroundingDINO_SwinB.cfg.py"
    cache_model = "path/to/your/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    dino = dino_model(model_config_path=cache_config_file, model_checkpoint_path=cache_model)
    dino.model.eval().cuda()

    # ================= 7. Determine output directory =================
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = ckpt_dir / f"{model_path.stem}_{timestamp}_{args.infer_set}"
    print(f"Output root: {output_root}")
    preds_dir = output_root / "preds"
    logits_dir = output_root / "logits"          # Reserved for CRF
    visuals_dir = output_root / "visuals"
    preds_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Log directory and file
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    infer_log = open(logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_infer.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, infer_log)
    sys.stderr = Tee(sys.stderr, infer_log)

    if args.infer_set == "test":
        raise ValueError("COCO has no test set!")
        test_result_dir = output_root / "raw" / "results" / "VOC2012" / "Segmentation" / "comp6_test_cls"
        test_result_dir.mkdir(parents=True, exist_ok=True)
        test_result_dir_crf = output_root / "crf" / "results" / "VOC2012" / "Segmentation" / "comp6_test_cls"
        test_result_dir_crf.mkdir(parents=True, exist_ok=True)

    # ================= 8. Visualization skeleton (all probes empty) =================
    # Visualization config override: fully controlled by command line
    if args.viz_eval_enabled:
        # Override training config with the probe list provided via command line
        if args.viz_probes and args.viz_probes.strip():
            cfg.visual.probes = [p.strip() for p in args.viz_probes.split(",") if p.strip()]
        else:
            cfg.visual.probes = []  # Visualization enabled but no probes specified, so clear
        # Validation strategy: trigger every step, unrestricted by training frequency
        cfg.visual.policy = {
            "warmup_steps": args.viz_warmup_steps,
            "warmup_freq": args.viz_warmup_freq,
            "epoch1_freq": args.viz_epoch1_freq,
            "later_freq": args.viz_later_freq,
        }
    else:
        cfg.visual.probes = []

    VISUALIZER.initialize(cfg, str(visuals_dir), fabric)

    # ================= 9. Inference loop =================
    # gts_list, preds_list = [], []
    # Use online confusion matrix instead of lists to avoid memory overflow
    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    images_path_list = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_data, desc="Inference", ncols=100)):
            # if idx > 10: break
            
            image, gt_mask, image_path, class_label, priors_payload = data

            img_name = Path(image_path[0]).stem

            # Check whether complete output already exists
            logit_file = logits_dir / f"{img_name}.npy"
            pred_file = preds_dir / f"{img_name}.png"
            if pred_file.exists():
                # Skip inference, directly load saved prediction mask for scoring
                pred = np.array(Image.open(pred_file), dtype=np.int16)
                # preds_list.append(pred)
                # gts_list.append(gt_mask[0])
                # Online accumulation of confusion matrix (validation set only)
                if args.infer_set == "val":
                    hist += _fast_hist(gt_mask[0].cpu().numpy().flatten(), pred.flatten(), args.num_classes)
                images_path_list.append(image_path[0])
                continue
            
            # Backpressure control: check logits pileup every 300 iters; if exceeding 600, wait
            if idx % 300 == 0 and args.crf_post:
                while len(list(logits_dir.glob("*.npy"))) > 2000:
                    print(f"[Inference] Excessive logits pileup, waiting for CRF processing... (current iter={idx})")
                    time.sleep(30)
                    
            VISUALIZER.update_state(epoch=0, iter=idx, mode='eval', silent=True)

            pseudo_label, r_mask = _infer_single(
                model, dino, cfg, args,
                image, gt_mask, image_path, class_label, priors_payload
            )

            img_name = Path(image_path[0]).stem

            images_path_list.append(image_path[0])
            if args.crf_post:
                logits_dir.mkdir(parents=True, exist_ok=True)
                np.save(logits_dir / f"{img_name}.npy", r_mask)

            if VISUALIZER.is_active("visualization_error_analysis"):
                # 1. Prepare GT (Tensor -> Numpy, remove Batch dimension)
                # gt_mask shape: [1, H, W] -> [H, W]
                curr_gt_mask = gt_mask[0].cpu().numpy().astype(np.uint8)
                
                # 2. Prepare filename
                img_name_stem = os.path.splitext(os.path.basename(image_path[0]))[0]
                
                # 3. Send to Visualizer (orig_img is RGB)
                orig_img = np.asarray(Image.open(image_path[0]))

                VISUALIZER.draw_semseg_error_analysis(
                    (orig_img, pseudo_label, curr_gt_mask),
                    img_name_stem
                )

            if args.infer_set == "val":
                # Save color-indexed mask
                pil_img = Image.fromarray(pseudo_label, mode='P')
                pil_img.putpalette(VOC_PALETTE.flatten().tolist())
                pil_img.save(preds_dir / f"{img_name}.png")
                # gts_list.append(gt_mask[0])
                # preds_list.append(pseudo_label)
                # Online accumulation of confusion matrix
                hist += _fast_hist(gt_mask[0].cpu().numpy().flatten(), pseudo_label.flatten(), args.num_classes)
                

            elif args.infer_set == "test":
                pass
                # # Test set submission format
                # pil_img = Image.fromarray(pseudo_label, mode='P')
                # pil_img.putpalette(VOC_PALETTE.flatten().tolist())
                # pil_img.save(test_result_dir / f"{img_name}.png")
           
        # ================= 10. Evaluation (val only) =================
        if args.infer_set == "val":
            # score = scores(gts_list, preds_list, n_class=args.num_classes, fabric=None)
            score = compute_metrics_from_hist(hist, args.num_classes)
            # If CRF is enabled, perform post-processing and evaluation
            
            with open(output_root / "results.json", "w") as f:
                json.dump(score, f, indent=4, sort_keys=True)
            print(f"Logits saved to {logits_dir}")
            print(f"Please run crf_only.py to perform CRF post-processing and compute CRF scores.")

        # ================= 11. Save run configuration snapshot =================
        with open(output_root / "run_args.txt", "w") as f:
            f.write("Command line: " + " ".join(sys.argv) + "\n")
            f.write(f"DINO threshold: box={cfg.box_threshold}, text={cfg.text_threshold}\n")
            f.write(f"Oracle filter: {cfg.get('eval_with_oracle_filter', False)}\n")
            f.write(f"Model path: {args.model_path}\n")
            f.write(f"Infer set: {args.infer_set}\n")

    print(f"Validation complete. Output directory: {output_root}")
    sys.exit(0)