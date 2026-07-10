# eval_utils.py
import os
from pathlib import Path
import torch
import lightning as L
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
import numpy as np
import torch.nn.functional as F
from PIL import Image
import json

from _Optimization_Workspace.tools.visualize import VISUALIZER, GRAY_SPEC, VIRIDIS_SPEC

palette = [ 128, 0, 0, 
           0, 128, 0, 
           128, 128, 0, 
           0, 0, 128, 
           128, 0, 128, 
           0, 128, 128, 
           128, 128, 128,
           64, 0, 0, 
           192, 0, 0, 
           64, 128, 0, 
           192, 128, 0, 
           64, 0, 128, 
           192, 0, 128, 
           64, 128, 128, 
           192, 128, 128,
           0, 64, 0, 
           128, 64, 0, 
           0, 192, 0, 
           128, 192, 0, 
           0, 64, 128,
           0, 0, 0]

dino_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background']

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
    
    

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


def _draw_validation_visuals(
    r_1024: torch.Tensor,
    image_path: list,
    gt_mask_orig: torch.Tensor,
    visualizer,
    priors_payload: dict,
    model_instance,
    raw_image_1024: torch.Tensor,
    cfg
):
    """
    [Core Diagnostic] Full-chain audit of the internal inference mechanism during validation.

    This function performs a comprehensive visualization review of the model's decision
    logic during the validation phase. It not only displays the final semantic segmentation
    results but also deeply probes the model's internal response to Hierarchical-Geometric
    Priors (NAMLab/Depth), and replicates the "Top1-Top2 decision-competition" probe from
    the loss function to evaluate alignment precision at boundaries.

    Visualization Dimensions:
    -------------------------
    1. Baseline: Logits heatmap, hard mask, GT, and input image in 1024 space.
    2. Raw Priors: Audit of raw physical prior boundaries (Audit_03/04).
    3. Processed Priors: Audit of the Prior-Guided Attractor Field processed internally (Audit_05).
    4. Hub Logits: Audit of model intermediate layer aggregation (Audit_06).
    5. Semantic Competition (Top2): Semantic Edge Probe replicated from the loss (Audit_07).
    6. Alignment Audit (Overlay): Red-green overlay of the Prior-Guided Attractor Field
       versus the decision-competition zone, intuitively reflecting alignment efficacy (Audit_08).

    Args:
        r_1024 (torch.Tensor): Aggregated 21-channel semantic canvas of shape [21, 1024, 1024].
        image_path (list): Disk path list for the current sample.
        gt_mask_orig (torch.Tensor): Ground-truth label map at original resolution [1, H_orig, W_orig].
        visualizer: Global VISUALIZER singleton.
        priors_payload (dict): Raw physical prior payload dictionary.
        model_instance: The model instance under validation (for extracting the processed_priors_dict repository).
        raw_image_1024 (torch.Tensor): Standardized 1024 RGB image before entering the model [1, 3, 1024, 1024].
        cfg (Box): Global experiment configuration object, used to extract loss weights and prior processing bills.
    """
    
    with torch.no_grad():
        sample_id = Path(image_path[0]).stem

        # --- Compute spatial padding parameters: align rectangular priors to SAM's 1024 square canvas ---
        def get_pad_cfg(image):
            oh, ow = image.shape[-2:]
            # oh, ow are the valid dimensions after scaling but before padding (e.g., 1024, 864)
            max_hw = max(oh, ow)
            oh = int(oh * 1024 / max_hw)
            ow = int(ow * 1024 / max_hw)
            # print(f"oh: {oh}, ow: {ow}")
            ph, pw = (1024 - oh) // 2, (1024 - ow) // 2
            # pad_cfg format: (left, right, top, bottom)
            pad_cfg = (pw, 1024 - ow - pw, ph, 1024 - oh - ph)
            return pad_cfg
        

        # --- Probe A: Baseline Monitoring (basic segmentation quality check) ---
        if visualizer.is_active('baseline_monitoring'):
            # 1. Generate 2D heatmap (max class probability after Softmax)
            # Only use corrected logits for heatmap computation, without affecting the original r_1024 for subsequent probes
            r_viz = r_1024.clone()
            for ch in range(1, r_viz.shape[0]):
                if r_viz[ch].eq(0).all():
                    r_viz[ch] = -100.0
            r_viz[0] = 0.0
            heatmap_2d = torch.softmax(r_viz, dim=0)[1:].max(dim=0).values
            visualizer.draw_heatmap(
                heatmap_2d, 
                tag="Val_Heatmap_1024", 
                title=f"Sample: {sample_id} - Logits Max"
            )
            
            # 2. Generate prediction mask (Argmax)
            pred_mask_2d = r_1024.argmax(dim=0)
            visualizer.draw_mask(
                pred_mask_2d, 
                tag="Val_Pred_1024", 
                title=f"Sample: {sample_id} - Prediction"
            )

            # 3. Prepare GT in 1024 space (for fair comparison)
            # Note: gt_mask_orig is [1, H_orig, W_orig]
            # We need to scale it to 1024 while using nearest-neighbor interpolation
            h_orig, w_orig = gt_mask_orig.shape[-2:]
            # The logic here should reference the Resize class to compute scaled h, w
            # Simplified: use F.interpolate to quickly align to 1024 space
            # Note: GT scaling MUST use nearest
            max_hw = max(h_orig, w_orig)
            gt_scaled = F.interpolate(
                gt_mask_orig.unsqueeze(0).float(), 
                size=(int(h_orig * 1024 / max_hw), int(w_orig * 1024 / max_hw)), 
                mode='nearest'
            )
            gt_1024 = F.pad(gt_scaled.squeeze(0), get_pad_cfg(gt_scaled.squeeze(0)), value=0).squeeze()
            
            visualizer.draw_mask(
                gt_1024, 
                tag="Val_GT_1024", 
                title=f"Sample: {sample_id} - GT (1024 Scale)"
            )

            # 4. Draw the corresponding input image
            visualizer.draw_image(
                F.pad(raw_image_1024[0], get_pad_cfg(raw_image_1024[0]), value=0),
                tag="Val_Input_1024", 
                title=f"Sample: {sample_id} - Input"
            )

        # --- Probe B: Top2 Edge Alignment (prior audit) ---
        if visualizer.is_active('top2_edge_alignment_monitoring'):
            processed_priors_dict = model_instance.processed_priors_dict
            logits_21ch = r_1024.unsqueeze(0).cuda()
            
            # 1. Raw prior audit (Raw Priors)
            if priors_payload:
                if 'namlab' in priors_payload:
                    # The namlab here is already at 1024 scale (after Resize)
                    # print(priors_payload['namlab'])
                    nam_raw = priors_payload['namlab'].float()
                    # Simple edge extraction for observation
                    nam_edges = (torch.nn.functional.max_pool2d(nam_raw, 3, 1, 1) != 
                                -torch.nn.functional.max_pool2d(-nam_raw, 3, 1, 1)).float()
                    visualizer.draw_binary_map(
                        F.pad(nam_edges.squeeze(), get_pad_cfg(nam_edges.squeeze()), value=0), 
                        tag="Val_Audit_NAMLab_Edges", 
                        title=f"Sample: {sample_id} - NAMLab Edges"
                    )
                if 'depth' in priors_payload:
                    d_tensor = priors_payload['depth'].float().squeeze()
                    d_norm = (d_tensor - d_tensor.min()) / (d_tensor.max() - d_tensor.min() + 1e-6)
                    visualizer.draw_heatmap(
                        F.pad(d_norm, get_pad_cfg(d_norm), value=0), 
                        tag="Val_Audit_Depth", 
                        title=f"Sample: {sample_id} - Depth", 
                        colormap_spec=GRAY_SPEC
                    )

            # 2. Processed prior audit (Processed Priors)
            if processed_priors_dict:
                for k, v in processed_priors_dict.items():
                    visualizer.draw_heatmap(
                        F.pad(v.squeeze(), get_pad_cfg(v.squeeze(),), value=0), 
                        tag=f"Val_Audit_Proc_{k}", 
                        title=f"Proc Prior: {k}", 
                        colormap_spec=VIRIDIS_SPEC
                    )

            # --- Deep Audit 07: Top1-Top2 Decision-Competition Zone (Semantic Edge Probe) ---
            # Replicated from losses.py: extract the class competition region at 1024 resolution
            # r_1024 is [21, 1024, 1024]; add a Batch dimension to fit the operator
            # Set all inactive regions (0.0) to extreme negative values, ensuring Top1 dominance after Softmax
            logits_clean = logits_21ch.clone()
            logits_clean[logits_21ch == 0] = -100.0
            probs_1024 = F.softmax(logits_clean, dim=1)
            probs_1024 = F.avg_pool2d(probs_1024, kernel_size=3, stride=1, padding=1)
            vals_1024, _ = torch.topk(probs_1024, k=2, dim=1)
            # Competition zone (1.0 represents the most intense competition)
            semantic_edge_1024 = (1.0 - (vals_1024[:, 0] - vals_1024[:, 1])).squeeze()

            visualizer.draw_heatmap(
                semantic_edge_1024, 
                tag="Audit_07_Semantic_Top2_Probe", 
                title=f"Semantic Competition (Top2): {sample_id}",
                colormap_spec=VIRIDIS_SPEC
            )

            # --- Deep Audit 08: Decision edge vs blurred physical edge (Alignment Audit) ---
            # Logic: only perform alignment audit when the corresponding loss term is enabled in config
            weights_cfg = cfg.loss.weights.get('top2_edge_alignment', {})
            
            bill_cfg = cfg.model.prior_configs.loss.top2_edge_alignment

            for p_type in ['namlab', 'depth']:
                w_key = f"logits_1024_{p_type}"
                if weights_cfg.get(w_key, 0.0) > 0:
                    # 1. Dynamic key name generation (strictly replicate losses.py algorithm)
                    req_tuple = bill_cfg.get(p_type)
                    # Strict audit: config requires but bill is missing, report logic error
                    if req_tuple is None:
                        raise KeyError(f"[Logic Error] {w_key} weight > 0, but no processed_requests in config!")
                    
                    processed_key = f"{p_type}_" + "_".join([f"{k}_{v}" for k, v in req_tuple])
                    
                    # 2. Physical extraction (extract from model repository processed_priors_dict)
                    # Strict audit: bill requires but repository is missing, report processing chain break error
                    if processed_key not in processed_priors_dict:
                        raise KeyError(f"[Process Error] Model repo missing expected key: '{processed_key}'. "
                                    f"Check model.prepare_for_image processing chain.")
                    
                    # 3. Draw comparison (Red: processed edge, Green: decision-competition edge)
                    # Extract and force-pad to 1024x1024, ensuring shape consistency with the semantic competition zone
                    target_edge_raw = processed_priors_dict[processed_key].squeeze()
                    target_edge_processed = F.pad(target_edge_raw, get_pad_cfg(target_edge_raw), value=0)
                    visualizer.draw_dual_channel_overlay(
                        data_r=target_edge_processed, 
                        data_g=semantic_edge_1024,
                        tag=f"Audit_08_{p_type}_Top2_Alignment", 
                        title=f"Red(Proc {p_type}) vs Green(Semantic) Alignment",
                        label_r=f"Physical Edge ({p_type}_processed)",
                        label_g="Semantic Competition (Top1-Top2)"
                    )


def validate(fabric: L.Fabric, cfg: Box, args, model: Model, dino_model1, val_dataloader: DataLoader, name: str, epoch: int = 0, save_dir=None):
    """
    Semantic segmentation validation core loop (Inference & Evaluation).

    This function simulates a single-stage inference pipeline:
    1. Use Grounding DINO to detect bounding boxes of objects in the image as spatial prompts.
    2. Input the bounding boxes into the SAM model to obtain instance mask logits on a 1024x1024 canvas.
    3. Based on DINO category indices, aggregate instance masks into a 21-channel semantic canvas
       via a Scatter-Max strategy.
    4. Physically anchor the background channel, then restore to the original image resolution
       via post-processing and compute metrics.

    Note (Heuristic Strategy):
    Before mask aggregation, this pipeline explicitly uses DINO classification confidence (logits)
    to weight SAM's raw mask logits — not pure logical competition, but overlaid with the detector's
    prior weights. This design aims to leverage the detector's semantic prior to moderate the
    influence of multi-instance overlapping regions and suppress low-confidence masks from false detections.

    Args:
        fabric (L.Fabric): Lightning Fabric distributed handle.
        cfg (Box): Global experiment configuration object.
        args: Command-line argument object, must include basic attributes such as num_classes.
        model (Model): Student network to be evaluated (SAM-based).
        dino_model1 (DINO): Pre-trained Grounding DINO model for generating Prompts.
        val_dataloader (DataLoader): Validation set loader, outputting single-batch data (B=1).
        name (str): Experiment name.
        epoch (int): Current training epoch.
        save_dir (str, optional): Save path for result logs and prediction maps.
    """
    model.eval()
    # Read validation mode from config, defaults to False (standard mode)
    use_oracle_filter = cfg.get("eval_with_oracle_filter", False)
    print(f"[Info] In eval_utils.py, validate, 📔 Using{'Oracle Filter mode, scores may be higher than standard⚠️' if use_oracle_filter else 'Standard mode🟢'}")
    fabric.print('======validation======')

    # Dynamically determine the output directory
    # If save_dir is not passed (e.g., legacy code logic), fall back to ./output/score (to prevent errors)
    json_save_dir = save_dir if save_dir else './output/score'
    os.makedirs(json_save_dir, exist_ok=True)

    preds_list, gts_list = [], []
    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            # --- Update global Visualizer state ---
            VISUALIZER.update_state(epoch=epoch, iter=iter + 1, mode='eval', silent=True)
            if iter % 20 == 0:
                print('eval_iter:', iter)

            # Data unpacking (Batch Size = 1)
            #   image: [1, 3, h, w] (valid image region after proportional scaling, without black border padding)
            #   gt_mask: [1, H_orig, W_orig] (original label map on disk)
            #   valid_size: (h, w) (valid resolution after scaling)
            #   image_path: List[str] (path list of length 1)
            #   class_label: [1, 21] (image-level multi-label classification Tensor)
            image, gt_mask, image_path, class_label, priors_payload = data

            # === [Core Injection] Let the model process and hold priors before inference (for auditing) ===
            model.prepare_for_image(priors_payload)
            # ======================================================

            valid_size = image.shape[-2:] # (h, w)

            dino_images = (image.squeeze(0).permute(1, 2, 0)*255).cpu().numpy().astype(np.uint8).copy()     # [h, w, 3] (Uint8, numpy, valid image region for DINO recognition)

            # Step 1: DINO object detection. Returns xyxy coordinates based on the [h, w] valid region.
            detections = dino_model1.predict_with_classes(
                dino_images, dino_class_names, cfg.box_threshold, cfg.text_threshold
            )                                               # DINO detection results. The xyxy coordinate system is based on [h, w].
            boxes = torch.tensor(detections.xyxy)           # [N, 4] (N is the number of detected instances)

            # --- DINO validation-side visualization probe ---
            if VISUALIZER.is_active("dino_boxes"):
                VISUALIZER.draw_detections(dino_images, detections, dino_class_names, tag="Dino_Boxes", title="Validation Image + DINO Boxes")
            # ----------------------------------
            
            # Step 2: Category ID mapping. Convert DINO (0-19) to VOC (1-20), background fixed as 0.
            class_ids = detections.class_id
            voc_class_ids = []
            for cid in class_ids:
                if cid == 20: # Background in DINO (if any)
                    voc_class_ids.append(0)
                else:
                    # DINO 0(aero) -> VOC 1(aero)
                    voc_class_ids.append(cid + 1)
            
            voc_class_ids = torch.tensor(voc_class_ids, dtype=torch.long)
            
            # Step 3: Initialize the 21-channel semantic canvas.
            # r: [21, 1024, 1024] (semantic confidence map for a single image at standard SAM resolution)
            r = torch.zeros((args.num_classes, 1024, 1024))
            
            # --- Coordinate system mapping ---
            # transformed_boxes: [N, 4] 
            # Logic: map DINO box coordinates based on [h, w] to the 1024 coordinate system required by the SAM encoder.
            # Since our [h, w] is already scaled with the longest side as 1024, the values here remain largely unchanged,
            # only performing internal format adaptation.
            transformed_boxes = model.apply_boxes_torch1(boxes, dino_images.shape[:2])

            mask_size = image.shape[-2:]
            height, width = gt_mask[0].shape

            if len(transformed_boxes) > 0:
                prompts = (transformed_boxes.cuda().to(dtype=torch.float64),)
                
                # Step 4: SAM encoding and decoding.
                #   dino_images_resize: [1, 3, h, w] (Float, 0-255).
                #   Note: model.forward internally auto-pads right/bottom to [1, 3, 1024, 1024] standard size.
                dino_images_resize = torch.tensor(dino_images).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                dino_images_resize = dino_images_resize.float() 
                
                #   pred_masks[0]: [N, 21, 1024, 1024] (instance-level logits).
                _, pred_masks, _, _ = model(dino_images_resize, prompts, (1024, 1024))
                
                # Overlay DINO confidence weights to reinforce accurately detected regions.
                masks = pred_masks[0] #!!!* logits[:, None, None].cuda()


                # Step 5: Semantic aggregation (Scatter-Max).
                # Iterate over all unique classes (1-20) detected in the current image.
                # r: [21, 1024, 1024] (21-class semantic canvas for a single image)
                # Logic: merge multiple prompt-box masks belonging to the same VOC category
                # into the corresponding channel via a max strategy.
                for unique_cid in voc_class_ids.unique():
                    # Skip prediction of the background class (0), if any
                    if unique_cid == 0: continue
                    
                    # Find all masks belonging to this category and fill r[unique_cid] with the maximum
                    # Note: r is [21, H, W]; directly indexing 1-20 is safe
                    class_masks = masks[voc_class_ids == unique_cid]
                    if len(class_masks) > 0:
                        r[unique_cid] = class_masks.max(dim=0)[0]
                
                # Step 6: Physically anchor the background channel (Index 0).
                # In regions where all foreground channels show no significant response (i.e., foreground_max <= 0),
                # set background confidence to 1.0.
                foreground_max = r[1:].max(dim=0).values
                
                # Where foreground shows no response, set the background channel to 1.0 (high confidence)
                # Where foreground does respond, keep the background channel at 0 (or set to a minimal value)
                r[0, foreground_max <= 0] = 1.0

                # --- [Visualize] Trigger validation audit visualization (perform mechanism audit in 1024 space) ---
                _draw_validation_visuals(
                    r_1024=r, 
                    image_path=image_path, 
                    gt_mask_orig=gt_mask, 
                    visualizer=VISUALIZER, 
                    priors_payload=priors_payload, 
                    model_instance=model, 
                    raw_image_1024=image, 
                    cfg=cfg
                )
                
                # Step 7: Post-processing. Crop black borders according to the valid region and interpolate/scale to the original image resolution.
                # r (input): [21, 1024, 1024] -> r (output): [21, H_orig, W_orig].
                r = sem_seg_postprocess(r, valid_size, height, width)
                # r_mask: [21, H_orig, W_orig] (fully restored to the original image scale)
                r_mask = r.cpu().numpy()

                # --- [Diagnostic Probe] Spatial alignment red-green overlay check ---
                if VISUALIZER.is_active('val_alignment_check'):
                    # 1. Extract predicted foreground (pixels with argmax > 0 are considered foreground)
                    # r is now [21, H_orig, W_orig]
                    pred_fg = (r.argmax(dim=0) > 0).float()
                    
                    # 2. Extract ground-truth foreground (gt_mask index 0 is typically background, > 0 is foreground)
                    # gt_mask shape is [1, H_orig, W_orig]
                    gt_fg = (gt_mask[0] > 0).float().to(pred_fg.device)
                    
                    # 3. Call red-green overlay visualization
                    # Extract the current image filename for identification
                    img_name = Path(image_path[0]).stem
                    VISUALIZER.draw_dual_channel_overlay(
                        data_r=gt_fg,       # Red: Ground Truth
                        data_g=pred_fg,      # Green: Prediction
                        tag=f"Val_Align_{img_name}",
                        title=f"Alignment Audit: {img_name}",
                        label_r="Ground Truth (FG)",
                        label_g="SAM Prediction (FG)"
                    )
                
                # ⚠️ Whether to use Oracle Filter evaluation mode
                if use_oracle_filter:
                    # Filter using GT labels (Oracle Filtering, original paper logic)
                    # Only keep categories present in GT + background
                    new_r_mask = np.zeros_like(r_mask)
                    
                    # Must preserve background (0)
                    new_r_mask[0] = r_mask[0] 
                    
                    # Iterate over categories present in GT (e.g., 1, 15)
                    # class_label[0] is tensor([0, 1, 0...])
                    # Note: eval_utils has label extraction logic under iter to ensure class_label is a tensor
                    active_classes = torch.nonzero(class_label[0]).squeeze(1)

                    for cls_idx in active_classes:
                        idx = cls_idx.item()
                        if idx > 0 and idx < 21: # 0 has already been handled
                            new_r_mask[idx] = r_mask[idx]
                    # Direct Argmax, this is the final label! No mapping needed!
                    # 0 is background, 1-20 are foreground, perfectly aligned
                    pseudo_label = np.argmax(new_r_mask, axis=0).astype(np.uint8)
                else:
                    # --- Mode B: Standard Inference (direct competition) ---
                    # Without looking at ground truth, whatever DINO+SAM predicts is what we use
                    new_r_mask = r_mask
                    pseudo_label = np.argmax(r_mask, axis=0).astype(np.uint8)
            else:
                # No Box situation, everything is background
                pseudo_label = np.zeros((height, width), dtype=np.uint8) # 全 0
                new_r_mask = np.zeros((args.num_classes, height, width), dtype=np.float32)
                new_r_mask[0, :, :] = 1.0

            orig_img = np.asarray(Image.open(image_path[0]))
            orig_img_ht = torch.from_numpy(orig_img)
            orig_img_ht = orig_img_ht.permute(2, 0, 1)
            orig_img_ht = orig_img_ht.unsqueeze(0)

            # Probe hook: error analysis
            # "visualization_error_analysis" must be enabled in the visual.probes list of config.py
            if VISUALIZER.is_active("visualization_error_analysis"):
                # 1. Prepare GT (Tensor -> Numpy, remove Batch dimension)
                # gt_mask shape: [1, H, W] -> [H, W]
                curr_gt_mask = gt_mask[0].cpu().numpy().astype(np.uint8)
                
                # 2. Prepare filename
                img_name_stem = os.path.splitext(os.path.basename(image_path[0]))[0]
                
                # 3. Send to Visualizer (orig_img is RGB)
                VISUALIZER.draw_semseg_error_analysis(
                    (orig_img, pseudo_label, curr_gt_mask),
                    img_name_stem
                )

            cam_wo_bg = new_r_mask[1:]    # [:args.background_class]
            cam_max = np.max(cam_wo_bg, (1,2), keepdims=True)
            cam_min = np.min(cam_wo_bg, (1,2), keepdims=True)
           
            predict = torch.from_numpy(pseudo_label).unsqueeze(0).numpy()

            preds_list += list(predict)
            gts_list += list(gt_mask[0][np.newaxis, :])

    score = scores(gts_list, preds_list, n_class=args.num_classes, fabric=fabric)

    if fabric.global_rank == 0:
        with open(json_save_dir + '_epoch' + str(epoch) + ('_oracle_filter' if use_oracle_filter else '_original') + f"{score['Mean IoU']:.4f}" + '.json', "w") as f:
            json.dump(score, f, indent=4, sort_keys=True)

    return score['Mean IoU']


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Restore semantic segmentation prediction results to the original image resolution.

    This function processes logits generated by a model (e.g., SAM) on a 1024x1024
    standard canvas. Since the input image undergoes proportional scaling and padding
    during preprocessing, this function is responsible for:
    1. Cropping the padded regions from the canvas based on the scaled valid dimensions (img_size).
    2. Bilinearly interpolating the cropped valid region back to the original disk resolution.

    Args:
        result (torch.Tensor): Semantic logits canvas output by the model, shape [C, 1024, 1024].
        img_size (tuple): Valid dimensions after scaling and before padding (h_scaled, w_scaled).
        output_height (int): Target output height (original image height H_orig).
        output_width (int): Target output width (original image width W_orig).

    Returns:
        torch.Tensor: Prediction probability map restored to original resolution,
            shape [C, output_height, output_width].
    """
    # Step 1: Compute padding offsets based on center alignment
    max_dim = 1024 
    pad_h = (max_dim - img_size[0]) // 2
    pad_w = (max_dim - img_size[1]) // 2
    
    # Step 2: Extract the valid prediction region (currently using center-crop logic)
    result = result[:, pad_h : pad_h + img_size[0], pad_w : pad_w + img_size[1]]

    # Step 3: Temporarily promote to 4D shape [1, C, H, W] to satisfy F.interpolate's input contract for 2D spatial interpolation
    result = result.unsqueeze(0)

    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


# File: utils/eval_utils.py 
from collections import defaultdict, deque
import torch

class SmoothedValue(object):
    """
    Track values of a sequence and provide smoothed averages (for log printing) and global averages (for final statistics).
    """
    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

class MetricLogger(object):
    """
    Intelligent logging steward.
    Automatically manages multiple SmoothedValue instances, supporting dynamic metric addition.
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, update_dict=None, n=1, **kwargs):
        """
        Update metrics.
        Args:
            update_dict: Dict-form metrics {name: value}
            n: Batch size (weight)
            **kwargs: Can also be passed via keyword arguments, e.g., update(loss=0.5, acc=0.9)
        """
        if update_dict is None:
            update_dict = {}
        update_dict.update(kwargs)
        
        for k, v in update_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'MetricLogger' object has no attribute '{}'".format(attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # Formatted print: Name [Median (GlobalAvg)]
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def averages(self):
        """Return the global average dictionary of all metrics (for TensorBoard)"""
        return {k: meter.global_avg for k, meter in self.meters.items()}

