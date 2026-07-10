# train_voc.py
from collections import defaultdict
import math
import sys
import os
from pathlib import Path
from typing import Optional
# Add workspace path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./_Optimization_Workspace")))
from tools.logger import setup_run

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import torch
import lightning as L
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch.nn as nn

try:
    from configs.config import cfg as default_cfg
except ImportError:
    default_cfg = None

from losses import Co2SAMCriterion
from datasets import call_load_dataset
from huggingface_hub import hf_hub_download
import argparse

from model import Model
from utils.eval_utils_voc import AverageMeter, validate, MetricLogger
from utils.tools import copy_model, momentum_update
from groundingdino.util.inference import Model as dino_model
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import supervision as sv

from _Optimization_Workspace.tools.visualize import VISUALIZER, GRAY_SPEC, VIRIDIS_SPEC
from _Optimization_Workspace.tools.lr_strategy import resolve_parameters_by_regex, LRStrategyManager
from _Optimization_Workspace.tools.gpu_monitor import GPU_MONITOR
from effsam_lora import fix_bn_states


dino_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'background']

class Co2SAMLRSchedule:
    """Serializable learning rate scheduling logic class, replacing local functions to support Checkpoint saving."""
    def __init__(self, cfg):
        self.warmup_steps = cfg.opt.warmup_steps
        self.steps = cfg.opt.steps
        self.decay_factor = cfg.opt.decay_factor

        self.schedule_type = cfg.opt.get("schedule_type", "step")
        self.poly_cfg = cfg.opt.get("poly", {})
        self.cosine_cfg = cfg.opt.get("cosine", {})
        self.gauss_cfg = cfg.opt.get("gaussian", {})

    def __call__(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        # Additional scheduling mode dispatch
        elif self.schedule_type == "poly":
            max_s, pwr, ratio = self.poly_cfg.get("max_steps", 1e5), self.poly_cfg.get("power", 0.9), self.poly_cfg.get("min_lr_ratio", 0.01)
            progress = min(1.0, (step - self.warmup_steps) / (max_s - self.warmup_steps))
            return ratio + (1 - ratio) * ((1 - progress) ** pwr)

        elif self.schedule_type == "cosine":
            max_s, ratio = self.cosine_cfg.get("max_steps", 1e5), self.cosine_cfg.get("min_lr_ratio", 0.01)
            progress = min(1.0, (step - self.warmup_steps) / (max_s - self.warmup_steps))
            return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        elif self.schedule_type == "gaussian":
            sigma = self.gauss_cfg.get("sigma")
            # Compute offset in absolute step count
            p = (step - self.warmup_steps) / sigma
            return math.exp(-(p**2))
        
        elif self.schedule_type == "step":
            # Original Step logic (as default fallback)
            if step < self.steps[0]:
                return 1.0
            elif step < self.steps[1]:
                return 1 / self.decay_factor
            else:
                return 1 / (self.decay_factor**2)
        else:
            raise ValueError("[Error] ❗ Incorrect schedule_type")

def _draw_chunk_heatmap_and_masks(
        s_masks, t_masks, detections, start, chunk_end, 
        image_path, gt_mask, visualizer, local_to_voc_ids,
        raw_image):
    """
    [Helper] Draw chunk-level aggregated heatmap.
    Logic derived from Part 1 in the original train.
    """
    with torch.no_grad():
        # 1. Create the overall canvas at the chunk level (fully consistent with validate logic)
        #    Get H, W. p_masks[0] is [K, H, W]
        _, H, W = s_masks[0].shape
        r_chunk = torch.zeros((21, H, W), device=s_masks[0].device)
        t_chunk = torch.zeros((21, H, W), device=s_masks[0].device)

        # 2. Prepare DINO information for the entire chunk
        chunk_detections = detections[start:chunk_end]
        chunk_logits_dino = torch.tensor(chunk_detections.confidence).cuda()
        
        voc_class_ids = []
        # DINO class_id is 0-19, map to VOC 1-20
        for cid in chunk_detections.class_id:
            voc_class_ids.append(local_to_voc_ids[cid])
        voc_class_ids = torch.tensor(voc_class_ids, dtype=torch.long).cuda()

        # 3. Compute weighted masks for the entire chunk
        s_masks_tensor = s_masks[0]

        best_mask_proposals = s_masks_tensor
        
        # [Dimensionality Audit] Confirm best_mask_proposals is 3D [N, H, W]
        if best_mask_proposals.ndim != 3:
            raise ValueError(f"CRITICAL: best_mask_proposals dimension is {best_mask_proposals.ndim}, expected 3D [N, H, W]. This indicates a model output change. Halting.")

        # [Core] Now the multiplication dimensions of weighted_masks are safe
        weighted_masks = best_mask_proposals * chunk_logits_dino[:, None, None]
        weighted_t_masks = t_masks[0] * chunk_logits_dino[:, None, None]
        
        # 4. Fill all foreground channels (iterative aggregation)
        for unique_cid in voc_class_ids.unique():
            class_masks = weighted_masks[voc_class_ids == unique_cid]
            if len(class_masks) > 0:
                # Core: take the maximum value of all masks belonging to this class and fill
                r_chunk[unique_cid] = class_masks.max(dim=0).values
            
            # Aggregate T-Teacher (t_masks)
            class_masks_t = weighted_t_masks[voc_class_ids == unique_cid]
            if len(class_masks_t) > 0:
                t_chunk[unique_cid] = class_masks_t.max(dim=0).values
        
        # 5. Uniformly generate the background channel (after all foreground channels are filled)
        foreground_max = r_chunk[1:].max(dim=0).values
        r_chunk[0, foreground_max <= 0] = 1.0
        
        # 6. Generate 2D heatmap for visualization
        # Logits -> Softmax -> 2D Probability Map (pixel-wise max confidence)
        heatmap_data = torch.softmax(r_chunk, dim=0).max(dim=0).values
        
        # Get sample ID for the title
        sample_id = Path(image_path[0]).stem
        
        VISUALIZER.draw_heatmap(
            heatmap_data,
            tag=f"Logits21ch_Heatmap_Aggregated", # Corrected tag, representing aggregated result
            title=f"Sample: {sample_id} - Aggregated"
        )

        # 7. Generate predicted mask based on aggregated result (Argmax logic)
        # r_chunk shape is [21, H, W], take max index along the class dimension
        pred_mask_2d = r_chunk.argmax(dim=0)
        
        # 8. Execute mask visualization drawing
        # Draw predicted mask
        visualizer.draw_mask(
            pred_mask_2d, 
            tag="Pred_Mask_Aggregated", 
            title=f"Sample: {sample_id} - Pred"
        )

        # 9. Draw T-Teacher (Template) mask
        t_pred_mask_2d = t_chunk.argmax(dim=0)
        visualizer.draw_mask(
            t_pred_mask_2d, 
            tag="Template_Mask_Aggregated", 
            title=f"Sample: {sample_id} - Template"
        )
        
        # Draw GT mask (take the current sample from the batch, gt_mask shape is typically [B, H, W])
        visualizer.draw_mask(
            gt_mask[0], 
            tag="GT_Mask", 
            title=f"Sample: {sample_id} - GT"
        )

        # 10. Draw original RGB image for alignment audit
        visualizer.draw_image(
            raw_image, 
            tag="Raw_RGB", 
            title=f"Sample: {sample_id} - RGB"
        )

def train(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    template_model: Model,
    ema_model: Model,  
    dino_model1: dino_model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    run_paths: Box, 
    dynamic_managers: Optional[list] = None
):
    """
    [Core Pipeline] Weakly Supervised Semantic Segmentation (WSSS) training main loop.
    
    Nested Loop Architecture:
    ---------------------------------------
    1. Outer Loop (Batch Level): 
       - Iteration unit: 1 Batch (current config B=1), representing one complete raw image
         and all its corresponding VOC category labels.
       - Core tasks: Execute DINO object detection, lock down the N potential object boxes
         (Prompts) across the entire image based on class_label.

    2. Inner Loop (Chunk Level):
       - Iteration unit: 1 Chunk (current config size=16).
       - Constraint: Since SAM memory consumption surges with the number of Prompts, the
         N objects in the entire image are split into multiple Chunks that pass through the
         model in batches.
       - State characteristic: The pred_masks output from a single inner loop iteration
         contain only 1/M of the local object instances in the entire image.

    3. Result Aggregation:
       - Completeness guarantee: Only when the inner loop (num_chunks) has fully executed,
         and the downstream visualization function (or Loss class) "assembles" all Chunk
         instance masks onto an 21-channel canvas according to the category indices provided
         by DINO, does the final semantic segmentation result for that sample take shape.

    Data Flow:
    -------------------
    - Input: images_weak (1024x1024) -> DINO (Prompts) -> Chunking (16 pts/group).
    - Forward: SAM Encoder -> SAM Decoder -> pred_masks [N, 1024, 1024].
    - Optimization: Accumulate Losses from all Chunks within the inner loop; trigger a
      unified parameter update from the main loop after the inner loop finishes.

    Args:
        cfg: Global configuration object.
        fabric: Lightning Fabric distributed training handle.
        model: Student model, responsible for gradient updates.
        template_model: Frozen teacher model (T-Teacher), providing initial pseudo-label reference.
        ema_model: Momentum-updated teacher model (U-Teacher), providing stable self-supervised signals.
        train_data / val_data: DataLoaders encapsulating images and priors (NAMLab/Depth).
    """
    criterion = Co2SAMCriterion(cfg)
    max_iou = 0.

    unfreeze_lora_epoch_iter = cfg.model.get("unfreeze_lora_epoch_iter", None)
    for epoch in range(1, cfg.num_epochs + 1):
        # --- Ensure training mode ---
        model.model.train()  # Enable LoRA training mode
        fix_bn_states(model, ema_model)
        template_model.eval() # Ensure Teacher is always in eval mode
        # ---------------------------
        batch_time = AverageMeter()
        data_time = AverageMeter()
        metric_logger = MetricLogger(delimiter="  ")
        end = time.time()
        num_iter = len(train_dataloader)


        for iter, data in enumerate(train_dataloader):
            try:
                if unfreeze_lora_epoch_iter:
                    if epoch == unfreeze_lora_epoch_iter[0] and iter == unfreeze_lora_epoch_iter[1]:
                        fabric.print(f"🚀 epoch {epoch}, iter {iter}, resetting optimizer and scheduler, unfreezing LoRA")
                        reset_optimizer_and_scheduler_for_lora(cfg, model, fabric)
                # --- Update global Visualizer state ---
                VISUALIZER.update_state(epoch=epoch, iter=iter + 1, mode='train', silent=True)
                # ------------------------------------

                data_time.update(time.time() - end)

                images_weak, images_strong, images, gt_mask, image_path, class_label, priors_payload = data

                # === [Logic Alignment] Process image-level background priors before the Chunk loop (B=1 logic) ===
                model.prepare_for_image(priors_payload)
                if ema_model is not None:
                    ema_model.prepare_for_image(priors_payload)
                # =================================================================
                # DataLoader outputs [0, 1], we need [0, 255] for SAM
                # Note: must generate new variables to avoid affecting DINO or other logic
                batched_input_weak = images_weak * 255.0
                batched_input_strong = images_strong * 255.0

                # --- Pure augmented image comparison probe (no DINO boxes) ---
                if VISUALIZER.is_active("aug_pairs"):
                    VISUALIZER.draw_image(images_weak[0], tag="Weak_Aug", title="Weak Augmentation Sample")
                    VISUALIZER.draw_image(images_strong[0], tag="Strong_Aug", title="Strong Augmentation Sample")
                # --------------------------------------------
                
                # 1. Take the first image from the batch
                dino_input_tensor = images_weak[0]  # shape: [3, 1024, 1024]
                
                # 2. Transpose dimensions: (C, H, W) -> (H, W, C)
                dino_input_np = dino_input_tensor.permute(1, 2, 0).cpu().numpy()
                
                # 3. Value conversion: Float(0-1) -> Uint8(0-255) and ensure memory contiguity
                dino_images = (dino_input_np * 255).astype(np.uint8)
                dino_images = np.ascontiguousarray(dino_images)

                prompts = torch.tensor([])

                # --- Correct Prompt generation logic ---
                # 1. Find all existing category indices for the current image (positions with value 1)
                # class_label[0] shape is [21]. Indices: 0=bg, 1=aero, ... 20=tv
                active_indices = torch.nonzero(class_label[0]).squeeze(1) 

                local_to_voc_ids = [idx.item() for idx in active_indices if idx.item() != 0]

                target_class_names = []

                # 2. If no valid foreground categories, skip subsequent processing
                detections = None # Explicit initialization

                for idx_tensor in active_indices:
                    idx = idx_tensor.item()
                    
                    # 3. Filter out background (index 0)
                    if idx == 0:
                        continue
                    
                    # 4. Map to DINO class name
                    # Dataset: 1=aeroplane ...
                    # DINO List: 0=aeroplane ...
                    # Relationship: dino_idx = dataset_idx - 1
                    dino_idx = idx - 1
                    
                    # Safety check
                    if dino_idx < 0 or dino_idx >= len(dino_class_names):
                        continue

                    target_class_names.append(dino_class_names[dino_idx]) # type: ignore
                    
                    

                if target_class_names:
                    # 3. Execute a single, unified DINO prediction
                    with GPU_MONITOR.scope("DINO_Inference"):
                        with torch.no_grad():
                            detections = dino_model1.predict_with_classes(
                                dino_images,
                                target_class_names, # Pass the list of all categories
                                cfg.box_threshold,
                                cfg.text_threshold
                            )

                    # --- Identity validity check and filtering ---
                    if detections.class_id is not None:
                        # Generate boolean mask: retain only valid detections where class_id is not None
                        valid_mask = np.array([cid is not None for cid in detections.class_id], dtype=bool)
                        
                        # Use boolean indexing to synchronously filter all fields: xyxy, confidence, class_id, etc.
                        detections = detections[valid_mask]
                        
                        assert detections is not None, "[Error] In train_voc.py train: detections cannot be None!"
                        # If detections are empty after filtering, explicitly reset to None to trigger skip logic
                        if len(detections) == 0:
                            detections = None
                    # ----------------------------------
                    # 5. Extract prompts from the unified detections object
                    if detections is not None and len(detections.xyxy) > 0: # type: ignore
                        prompts = torch.tensor(detections.xyxy).to(device=fabric.device, dtype=torch.float64) # type: ignore

                        # --- DINO input visualization probe ---
                        if VISUALIZER.is_active("dino_boxes"):
                            VISUALIZER.draw_detections(images_weak[0], detections, target_class_names, tag="Dino_Original", title="Original Image + DINO Boxes")
                        # ----------------------------------
                        

                # ----------------- Memory optimization: timely release of large arrays -----------------
                del dino_images
                # ---------------------------------------------------------


                if len(prompts) > 0:
                    all_prompts = prompts.to(device=fabric.device, dtype=torch.float64)
                    total_num_masks = len(all_prompts)
                    
                    # Initialize logging variables
                    batch_metrics = defaultdict(float) # Requires from collections import defaultdict
                    
                    chunk_size = 16 
                    num_chunks = int(np.ceil(total_num_masks / chunk_size))

                    for chunk_idx in range(num_chunks):
                        start = chunk_idx * chunk_size
                        chunk_end = min((chunk_idx + 1) * chunk_size, total_num_masks)
                        
                        # Construct Chunk
                        current_prompts_tensor = all_prompts[start:chunk_end]
                        chunk_prompts = (current_prompts_tensor,) 

                        with GPU_MONITOR.scope(f"SAM_Chunk_{chunk_idx}_Forward"):
                            # 1. Forward propagation (back inside the loop, ensuring memory safety)
                            # Although each round requires encoding, the total number of encodes is greatly reduced
                            # because chunk_size is larger
                            with torch.no_grad():
                                t_embeds, t_masks, t_iou, t_res = template_model(batched_input_weak, chunk_prompts, images_strong.shape[-2:])
                            
                            # U-Teacher inference
                            # If EMA is enabled, use ema_model; otherwise fall back to model (Siamese)
                            u_teacher = ema_model if ema_model is not None else model
                            with torch.no_grad():
                                s_embeds, s_masks, s_iou, s_res = u_teacher(batched_input_weak, chunk_prompts, images_strong.shape[-2:])

                            p_embeds, p_masks, p_iou, p_res = model(batched_input_strong, chunk_prompts, images_strong.shape[-2:])

                        # === Extract VOC category IDs for the current Chunk ===
                        if detections:
                            assert detections, "[Error] In train_voc.py train: detections cannot be None!"
                            chunk_detections = detections[start:chunk_end]
                            assert isinstance(chunk_detections, sv.Detections), "[Error] In train_voc.py train: chunk_detections cannot be None!" # type: ignore
                            # print(chunk_detections.class_id, type(chunk_detections.class_id))
                            assert isinstance(chunk_detections.class_id, np.ndarray), "[Error] In train_voc.py train: chunk_detections cannot be None!"
                            curr_chunk_voc_ids = torch.tensor(
                                [local_to_voc_ids[cid] for cid in chunk_detections.class_id], 
                                dtype=torch.long, device=p_masks[0].device
                            )
                        else:
                            curr_chunk_voc_ids = None
                        # ==========================================

                        # 2. Compute Loss
                        chunk_loss_acc = 0.0
                        
                        for i in range(len(p_masks)):
                            # 1. Prepare single-sample inputs (handle Batch dimension)
                            curr_soft_embed = s_embeds[i] if len(s_embeds) > i else s_embeds[0]
                            curr_temp_embed = t_embeds[i] if len(t_embeds) > i else t_embeds[0]

                            # 2. Extract model internal products for the current sample (unpack from Hub)
                            curr_hub = {k: v[i] if isinstance(v, list) else v for k, v in model.output_hub.items()}
                            
                            processed_priors_dict = model.processed_priors_dict
                            
                            # 3. Invoke the new module (automatically compute and return dict)
                            with GPU_MONITOR.scope("Criterion_Evaluation"):
                                loss_dict = criterion(
                                    pred_mask=p_masks[i],
                                    soft_mask=s_masks[i],
                                    template_mask=t_masks[i],
                                    soft_embed=curr_soft_embed,
                                    temp_embed=curr_temp_embed,
                                    soft_res=s_res[i].clone().detach(),
                                    temp_res=t_res[i].clone().detach(),
                                    total_num_masks=total_num_masks,
                                    output_hub = curr_hub,
                                    priors_payload=priors_payload,
                                    processed_priors=processed_priors_dict,
                                    chunk_voc_ids=curr_chunk_voc_ids,
                                    epoch=epoch, iter=iter,
                                )

                            
                            # 4. Unpack variables (keep compatibility with logging code below)
                            step_loss = loss_dict["total"]
                            
                            chunk_loss_acc += step_loss

                            # Log accumulation
                            # Automatically accumulate all returned Loss values
                            for k, v in loss_dict.items():
                                if isinstance(v, torch.Tensor):
                                    v = fabric.all_reduce(v, reduce_op="mean").item()
                                batch_metrics[k] += v

                        # 6. Backward pass and release graph
                        with GPU_MONITOR.scope("Fabric_Backward"):
                            fabric.backward(chunk_loss_acc) # type: ignore
                            if cfg.opt.get("clip_grad"):
                                fabric.clip_gradients(model, optimizer, max_norm=cfg.opt.clip_grad)
                        
                    # --- [Branch A] Baseline RGB images, masks, and heatmaps
                    if VISUALIZER.is_active('baseline_monitoring'):
                        _draw_chunk_heatmap_and_masks(s_masks, t_masks, detections, start, chunk_end, image_path, gt_mask, VISUALIZER, local_to_voc_ids, images[0])


                    # --- [Branch B] Full-pipeline prior and intermediate product audit (Prior Audit) ---
                    if VISUALIZER.is_active('top2_edge_alignment_monitoring'):
                        # 1. Raw prior audit (Raw Priors)
                        if priors_payload:
                            if 'namlab' in priors_payload:
                                nam_raw = priors_payload['namlab'].float()
                                nam_edges = (torch.nn.functional.max_pool2d(nam_raw, 3, 1, 1) !=  # type: ignore
                                            -torch.nn.functional.max_pool2d(-nam_raw, 3, 1, 1)).float() # type: ignore
                                VISUALIZER.draw_binary_map(nam_edges.squeeze(0), tag="Audit_03_Raw_NAMLab", title="NAMLab Boundaries")
                            if 'depth' in priors_payload:
                                d_tensor = priors_payload['depth'].float()
                                d_min, d_max = d_tensor.min(), d_tensor.max()
                                d_norm = (d_tensor - d_min) / (d_max - d_min + 1e-6)
                                VISUALIZER.draw_heatmap(d_norm.squeeze(0), tag="Audit_04_Raw_Depth", title="Normalized Depth (0-1)", colormap_spec=GRAY_SPEC)

                        # 2. Processed prior audit (Processed Priors - automatically iterate)
                        if processed_priors_dict:
                            for k, v in processed_priors_dict.items():
                                # v shape is [1, 1024, 1024], reduce to 2D via squeeze
                                VISUALIZER.draw_heatmap(v.squeeze(), tag=f"Audit_05_Proc_{k}", title=f"Processed Prior: {k}", colormap_spec=VIRIDIS_SPEC)

                        # 3. Model intermediate product audit (Hub Logits - aggregated observation)
                        for hub_key in ['logits_256', 'logits_1024']:
                            if hub_key in curr_hub:
                                # curr_hub[hub_key] shape is [N, 1, H, W]; take max along N dimension for semantic aggregation
                                agg_logits = curr_hub[hub_key].max(dim=0).values.squeeze()
                                VISUALIZER.draw_heatmap(agg_logits, tag=f"Audit_06_Hub_{hub_key}", title=f"Aggregated {hub_key} (Interior)")

                    del detections

                    # Update parameters
                    optimizer.step()
                    scheduler.step()

                    # --- Dynamic learning rate manager mount point ---
                    if dynamic_managers:
                        for manager in dynamic_managers:
                            manager.update(batch_metrics, optimizer)
                    # ----------------------------------
                    
                    optimizer.zero_grad()
                    # Update EMA
                    if ema_model is not None:
                        unwrapped_model = model.module if hasattr(model, "module") else model
                        momentum_update(unwrapped_model, ema_model, cfg.ema_rate)

                    # === Clean up mounts to prevent memory leaks ===
                    model.current_priors = None
                    model.processed_priors_dict.clear()
                    if ema_model is not None:
                        ema_model.current_priors = None
                        ema_model.processed_priors_dict.clear()
                    # ======================================


                    torch.cuda.empty_cache()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    # Update Metrics
                    batch_size = images_weak.size(0)
                    # 1. Update global steward
                    metric_logger.update(batch_metrics, n=batch_size)
                    
                    # 2. Print (auto string concatenation)
                    # By default, metric_logger values are: `metric: median of last 20 iters (global average)`
                    # See utils.eval_utils.MetricLogger and SmoothedValue for details
                    fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                            f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                            f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                            f' | {str(metric_logger)}') # <--- Automatically expands all Loss values here

                    # 3. Record (auto-generate dict)
                    fabric.log_dict(metric_logger.averages(), num_iter * (epoch - 1) + iter)

                    # Periodic synchronized metrics snapshot (Decoupled Sync)
                    # Uses the '@all' macro directive, relying only on sampling frequency,
                    # without checking specific probe whitelists.
                    # Ensures any modifications to VISUALIZER.metrics in the current iteration are persisted.
                    if VISUALIZER.is_active('@all'):
                        VISUALIZER.sync()

                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if GPU_MONITOR.is_oom_exception(e):
                    fabric.print(GPU_MONITOR.generate_autopsy_report())
                raise e
            

            if iter % 50 == 0:
                print(GPU_MONITOR.get_status_line())

        if epoch % cfg.eval_interval == 0:
            iou = validate(fabric, cfg, args, ema_model, dino_model1, val_dataloader, 
                           cfg.name, epoch, save_dir=run_paths.log if run_paths else None)
            
            unwrapped_model = model.module if hasattr(model, "module") else model
            unwrapped_u_teacher_model = ema_model.module if hasattr(ema_model, "module") else ema_model
            student_state = {"model": unwrapped_model, "optimizer": optimizer, "scheduler": scheduler}
            u_teacher_state = {"model": unwrapped_u_teacher_model}
            student_save_path = os.path.join(run_paths.ckpt, f"student_model_epoch_{epoch}_miou_{iou:.4f}.pth")
            u_teacher_save_path = os.path.join(run_paths.ckpt, f"u_teacher_model_epoch_{epoch}_final_miou_{iou:.4f}.pth")
            
            fabric.save(student_save_path, student_state)
            fabric.save(u_teacher_save_path, u_teacher_state)
            if iou > max_iou:
                fabric.print(f"★ Saved Best Checkpoint (IoU: {iou:.4f}): {u_teacher_save_path}")
                max_iou = iou


def configure_opt(cfg: Box, model: Model, fabric: L.Fabric):
    lr_lambda = Co2SAMLRSchedule(cfg)

    if cfg.model.get('set_all_params_frozen', False):   # If True, freeze everything
        for param in model.model.parameters():
            param.requires_grad = False
        print("[Config] set_all_params_frozen is True. All parameters frozen initially.")
    else:
        # 1. Force-freeze Prompt Encoder and Mask Decoder (explicitly required by the paper)
        for param in model.model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in model.model.mask_decoder.parameters():
            param.requires_grad = False

        # 2. Intelligent handling of Image Encoder (train only LoRA)
        # Note: cannot directly freeze model.model.image_encoder.parameters(), or it will freeze LoRA too
        lora_count = 0
        backbone_count = 0
        
        for name, param in model.model.image_encoder.named_parameters():
            if "linear_a" in name or "linear_b" in name:
                # This is a LoRA parameter, must be trained
                param.requires_grad = True
                lora_count += 1
            else:
                # This is a ViT backbone parameter, must be frozen
                param.requires_grad = False
                backbone_count += 1

        
    # [Targeted unfreeze 1] If Resize-Convolution is enabled, must unfreeze new parameters of the upscaling layers
    patches = cfg.model.get('patches', {})
    rc_frozen = cfg.model.freeze.get('resize_convolution', False)
    if patches.get('mask_decoder_upscaling') == 'resize_convolution':
        if not rc_frozen:
            fabric.print("[Config] Resize-Convolution detected. Unfreezing upscaling layers in MaskDecoder.")
            # Precisely unfreeze Conv2d layers (newly added layers)
            for module in model.model.mask_decoder.output_upscaling.modules():
                if isinstance(module, torch.nn.Conv2d):
                    for param in module.parameters():
                        param.requires_grad = True
        else: fabric.print("[Config] ⚠️ Resize-Convolution detected, but set to frozen🥶")
        

    # [Targeted unfreeze 2] If the decoder uses the original CT,
    if patches.get('mask_decoder_upscaling', None) == None \
        and cfg.model.freeze.get('ConvTranspose', True) == False: # Default is kept frozen, i.e. True
        # Precisely unfreeze Conv2d layers (newly added layers)
        fabric.print("[Config] 🔥 Unfreezing CT:")
        for module in model.model.mask_decoder.output_upscaling.modules():
            if isinstance(module, torch.nn.ConvTranspose2d):
                for param in module.parameters():
                    param.requires_grad = True
                fabric.print(f"- Unfrozen module: {module}")
    

    

    # 3. Collect all parameters with requires_grad=True
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    
    
    if len(trainable_params) == 0:
        raise ValueError("FATAL: No parameters are trainable! LoRA injection might have failed.")
    
    # --- Dynamic learning rate parameter grouping logic ---
    dynamic_managers = []
    dynamic_param_ids = set()
    param_groups = []

    # Get dynamic config (cfg.opt.dynamic_schemes)
    # Structure example: {'template': {'output_upscaling': {'rates': {...}, 'allow_bounce': False}}}
    dynamic_schemes = cfg.opt.get('dynamic_schemes', {})
    
    for loss_key, locators in dynamic_schemes.items():
        for locator_key, scheme_cfg in locators.items():
            # 1. Structural regex parsing and auditing (failure triggers immediate error halt)
            target_params = resolve_parameters_by_regex(model.model, locator_key)
            
            # 2. Conflict audit: check whether parameters have already been assigned
            for p in target_params:
                if id(p) in dynamic_param_ids:
                    raise RuntimeError(f"Conflict: Parameter in {locator_key} already assigned to another dynamic scheme.")
                dynamic_param_ids.add(id(p))
            
            # 3. Create independent parameter group
            group_idx = len(param_groups) + 1 # Group 0 is usually the base group
            param_groups.append({
                'params': target_params,
                'lr': cfg.opt.learning_rate # Initial 1x
            })
            
            # 4. Register manager
            manager = LRStrategyManager(group_idx, loss_key, scheme_cfg)
            dynamic_managers.append(manager)
            fabric.print(f"[Config] Dynamic LR scheme activated: {loss_key} -> {locator_key}")

    # 5. Collect remaining parameters (Base Group)
    base_params = [p for p in trainable_params if id(p) not in dynamic_param_ids]
    final_groups = [{'params': base_params, 'lr': cfg.opt.learning_rate}] + param_groups
    
    opt_type = cfg.opt.get("type", "adam")
    if opt_type == "adam":
        optimizer = torch.optim.Adam(final_groups, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay) # type: ignore
        fabric.print("[Info] Using Adam optimizer")
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(final_groups, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)  # type: ignore
        fabric.print("[Info] Using AdamW optimizer")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fix_bn_states(model)

    return optimizer, scheduler, dynamic_managers


def reset_optimizer_and_scheduler_for_lora(
    cfg: Box,
    model: Model,
    fabric: L.Fabric
):
    """
    Reset optimizer and scheduler mid-training, specifically for the set_all_params_frozen=True
    startup strategy.
    This function maintains absolute consistency with the core logic in configure_opt,
    while forcibly unfreezing LoRA parameters.
    """    
    # 1. Force-freeze Prompt Encoder and Mask Decoder (explicitly required by the paper)
    for param in model.model.prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.model.mask_decoder.parameters():
        param.requires_grad = False

    # 2. Intelligent handling of Image Encoder (train only LoRA)
    # Note: cannot directly freeze model.model.image_encoder.parameters(), or it will freeze LoRA too
    lora_count = 0
    backbone_count = 0
    
    for name, param in model.model.image_encoder.named_parameters():
        if "linear_a" in name or "linear_b" in name:
            # This is a LoRA parameter, must be trained
            param.requires_grad = True
            lora_count += 1
        else:
            # This is a ViT backbone parameter, must be frozen
            param.requires_grad = False
            backbone_count += 1

    # [Targeted unfreeze 1] If Resize-Convolution is enabled, must unfreeze new parameters of the upscaling layers
    patches = cfg.model.get('patches', {})
    rc_frozen = cfg.model.freeze.get('resize_convolution', False)
    if patches.get('mask_decoder_upscaling') == 'resize_convolution':
        if not rc_frozen:
            fabric.print("[Config] Resize-Convolution detected. Unfreezing upscaling layers in MaskDecoder.")
            # Precisely unfreeze Conv2d layers (newly added layers)
            for module in model.model.mask_decoder.output_upscaling.modules():
                if isinstance(module, torch.nn.Conv2d):
                    for param in module.parameters():
                        param.requires_grad = True
        else: 
            fabric.print("[Config] ⚠️ Resize-Convolution detected, but set to frozen🥶")
        
    # [Targeted unfreeze 2] If the decoder uses the original CT,
    if patches.get('mask_decoder_upscaling', None) is None \
        and cfg.model.freeze.get('ConvTranspose', True) is False: # Default is kept frozen, i.e. True
        # Precisely unfreeze Conv2d layers (newly added layers)
        fabric.print("[Config] 🔥 Unfreezing CT:")
        for module in model.model.mask_decoder.output_upscaling.modules():
            if isinstance(module, torch.nn.ConvTranspose2d):
                for param in module.parameters():
                    param.requires_grad = True
                fabric.print(f"- Unfrozen module: {module}")

    # 3. Collect all parameters with requires_grad=True
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        raise ValueError("FATAL: No parameters are trainable! LoRA injection might have failed.")
    
    # --- Dynamic learning rate parameter grouping logic (maintain absolute consistency with configure_opt) ---
    dynamic_managers = []
    dynamic_param_ids = set()
    param_groups = []

    dynamic_schemes = cfg.opt.get('dynamic_schemes', {})
    
    for loss_key, locators in dynamic_schemes.items():
        for locator_key, scheme_cfg in locators.items():
            # 1. Structural regex parsing and auditing
            target_params = resolve_parameters_by_regex(model.model, locator_key)
            
            # 2. Conflict audit
            for p in target_params:
                if id(p) in dynamic_param_ids:
                    raise RuntimeError(f"Conflict: Parameter in {locator_key} already assigned to another dynamic scheme.")
                dynamic_param_ids.add(id(p))
            
            # 3. Create independent parameter group
            group_idx = len(param_groups) + 1
            param_groups.append({
                'params': target_params,
                'lr': cfg.opt.learning_rate
            })
            
            # 4. Register manager
            manager = LRStrategyManager(group_idx, loss_key, scheme_cfg)
            dynamic_managers.append(manager)
            fabric.print(f"[Config] Dynamic LR scheme activated: {loss_key} -> {locator_key}")

    # 5. Collect remaining parameters (Base Group)
    base_params = [p for p in trainable_params if id(p) not in dynamic_param_ids]
    final_groups = [{'params': base_params, 'lr': cfg.opt.learning_rate}] + param_groups
    
    # --- Optimizer and scheduler instantiation (maintain absolute consistency with configure_opt) ---
    opt_type = cfg.opt.get("type", "adam")
    if opt_type == "adam":
        optimizer = torch.optim.Adam(final_groups, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
        fabric.print("[Info] Using Adam optimizer")
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(final_groups, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
        fabric.print("[Info] Using AdamW optimizer")

    # Note: must re-instantiate lr_lambda here to ensure the scheduler starts counting from 0
    lr_lambda = Co2SAMLRSchedule(cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Fix BN state (aligned with initialization)
    fix_bn_states(model)

    # Return new instances
    return optimizer, scheduler, dynamic_managers

def enforce_strict_determinism(seed):
    """
    Force the use of deterministic operators.
    """
    # 1. Eliminate low-level C++ and Python hash randomness
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Force cuBLAS matrix multiplication determinism (for CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # 3. Force PyTorch to use deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True) # type: ignore
    
    # 4. Force cuDNN (convolution backend library) to use deterministic algorithms and disable auto-tuning
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore

def main(cfg: Box, args) -> None:
    if cfg.rand_seed is not None and cfg.get("determinism", True):
        enforce_strict_determinism(cfg.rand_seed)
        print(f"[Info] ✅ Random seed: {cfg.rand_seed}, deterministic operators enforced")
    else:
        print("[Warning] ⚠️ Deterministic operators not enforced")

    # 1. Initialize experiment directory (with config parameter)
    run_paths = None
    if args.config:
        run_paths = setup_run(args.config, cfg.get('output_dir_name', ""))
        # Inject paths into cfg for compatibility
        cfg.run_paths = run_paths 

    
    # 2. Configure Fabric to point to the new log directory
    # Note: name=None, version=None ensures direct writing to the log directory without creating subfolders
    loggers = []
    if run_paths:
        loggers.append(TensorBoardLogger(root_dir=run_paths.log, name=None, version=None))
    else:
        # Fallback logic (if no run_paths, use the original)
        loggers.append(TensorBoardLogger(cfg.out_dir))
    
    gpu_ids = cfg.gpu_ids.split(',')
    print(f"[Info] ⭐ GPUs used: {gpu_ids}")
    GPU_MONITOR.set_gpu_ids(gpu_ids=[int(i) for i in gpu_ids])
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=loggers)
    fabric.launch()
    
    if cfg.rand_seed is not None:
        fabric.seed_everything(cfg.rand_seed + fabric.global_rank)
        print("[Info] ✅ fabric.seed_everything")
    else:
        print("[Warning] ⚠️ fabric.seed_everything not enabled")

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data = load_datasets(cfg, cfg.model_img_size or 1024, fabric)

    # --- Initialize the Global Visualizer ---
    if run_paths:
        VISUALIZER.initialize(cfg, run_paths.viz, fabric)
    # ----------------------------------------
    
    # ================= [Architecture Lifecycle] =================
    # 1. Snapshot T-Teacher (Standard Structure & Pretrained Weights)
    #    Must copy before patching to ensure the teacher holds the original structure
    template_model = copy_model(model).to(fabric.device)
    GPU_MONITOR.register_entity("T-teacher", template_model)
    print("[Info] Template Model initialized (Standard Structure).")

    # 2. Patch Student (Apply Resize-Conv, etc.)
    #    Must execute before Optimizer initialization, otherwise new parameters cannot enter the optimizer
    model.apply_structure_patches()
    # ============================================================

    optimizer, scheduler, dynamic_managers = configure_opt(cfg, model, fabric)

    GPU_MONITOR.register_entity("Optimizer", optimizer)

    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    
    GPU_MONITOR.register_entity("Student", model)

    # ----------------- Completely freeze Template Model -----------------
    template_model.eval()  # Disable Dropout/BatchNorm statistics
    for param in template_model.parameters():
        param.requires_grad = False # Explicitly cut off gradients to prevent unintended computation graph attachment
    # -----------------------------------------------------------------

    model, optimizer = fabric.setup(model, optimizer)

    
    if cfg.resume:
        if cfg.model.ckpt is None:
            raise ValueError("[in train_voc.py] When using resume, ckpt cannot be None!")
        
        u_teacher_full_checkpoint = fabric.load(cfg.model.ckpt)
        
        ema_model = None
        if cfg.ema_rate > 0:
            ema_model = copy_model(model).to(fabric.device)
            unwrapped_u_teacher_model = ema_model.module if hasattr(ema_model, "module") else ema_model
            unwrapped_u_teacher_model.load_state_dict(u_teacher_full_checkpoint["model"], strict=False) # type: ignore
            ema_model.eval()
            for param in ema_model.parameters():
                param.requires_grad = False
            GPU_MONITOR.register_entity("U-teacher", ema_model)
            print(f"[Info] EMA Model enabled with rate: {cfg.ema_rate}")
        else:
            print("[Info] EMA Model disabled (Siamese mode)")
            
        # === Unwrap model to ensure 100% key alignment between loading and saving ===
        unwrapped_model = model.module if hasattr(model, "module") else model

        if cfg.model.stu_ckpt is not None:
            if cfg.ema_rate > 0:
                student_full_checkpoint = fabric.load(cfg.model.stu_ckpt)
                fabric.print("✅ Loaded Student weight checkpoint")
            else:
                raise ValueError("❌ Want to use both Student and U-teacher checkpoints, but EMA is disabled?")
        else:
            student_full_checkpoint = fabric.load(cfg.model.ckpt)
            fabric.print("📖 No Student weight checkpoint detected, loading U-teacher's instead!")
        
        # Force capture of missing key names to break silent failures
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(student_full_checkpoint["model"], strict=False)
        
        print(f"[Config] Using checkpoints: U-teacher: {cfg.model.ckpt}\nStudent: {cfg.model.stu_ckpt}")
        
        # If there are still unloaded keys, forcefully print them—never let it happen silently again
        if missing_keys:
            print(f"[Error/Warning] 🚨 Critically missing parameter keys (not successfully loaded): {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] ⚠️ Redundant parameter keys in checkpoint: {unexpected_keys}")

        if cfg.resume_opt:
            if cfg.model.stu_ckpt is None:
                raise ValueError("❌ Student weight path not provided, cannot use checkpoint's optimizer and scheduler!")
            optimizer.load_state_dict(student_full_checkpoint["optimizer"])
            # Compatibility handling: check whether the loaded object is an object or a state dict
            saved_scheduler = student_full_checkpoint["scheduler"]
            if isinstance(saved_scheduler, dict):
                scheduler.load_state_dict(saved_scheduler)
            else:
                # If it is the object itself, extract its state dict
                scheduler.load_state_dict(saved_scheduler.state_dict())
            print("  - ✅️ Using saved optimizer and scheduler parameters")
        else:
            print("  - ⚠️ Not using saved optimizer and scheduler parameters")
    else:
        # ===  Initialize EMA Model (U-Teacher) ===
        ema_model = None
        if cfg.ema_rate > 0:
            ema_model = copy_model(model).to(fabric.device)
            ema_model.eval()
            for param in ema_model.parameters():
                param.requires_grad = False
            GPU_MONITOR.register_entity("U-teacher", ema_model)
            print(f"[Info] EMA Model enabled with rate: {cfg.ema_rate}")
        else:
            print("[Info] EMA Model disabled (Siamese mode)")

    print(f"[Info] ✅ Learning rate: {cfg.opt}")


    # Initialize grounding DINO
    repo_id = "ShilongLiu/GroundingDINO"
    filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    # Method 1: Download online
    # cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    # cache_model = hf_hub_download(repo_id=repo_id, filename=filename)

    # Method 2: Local reading
    cache_config_file = "path/to/your/GroundingDINO/GroundingDINO_SwinB.cfg.py"
    cache_model = "path/to/your/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
    dino_model1 = dino_model(model_config_path=cache_config_file,
                                   model_checkpoint_path=cache_model)
    dino_model1.model = fabric.setup(dino_model1.model)
    dino_model1.model.eval()

    GPU_MONITOR.register_entity("DINO", dino_model1) # type: ignore

    eval_before_training = cfg.get("eval_before_training", True)
    if eval_before_training:
        # --- Logic routing: determine the evaluation target object before training ---
        # If resume is configured and the path exists, validate the loaded Student, otherwise validate the original Teacher
        if cfg.resume and cfg.model.ckpt is not None:
            eval_target = model
            fabric.print(f"[Info] 🚀 Pre-training evaluation: Pre-trained weights detected, using student model (Student) to verify loading. Path: {cfg.model.ckpt}")
        else:
            eval_target = template_model
            fabric.print("[Info] 🛡️ Pre-training evaluation: No weight loading detected, using template model (T-Teacher) as initial baseline.")
        
        validate(fabric, cfg, args, eval_target, dino_model1, val_data, name=cfg.name, epoch=0, save_dir=run_paths.log if run_paths else None) # type: ignore
    train(cfg, fabric, model, template_model, ema_model, dino_model1, optimizer, scheduler, train_data, val_data, run_paths=run_paths, dynamic_managers=dynamic_managers) # type: ignore

    del model, template_model, train_data, val_data

    
def pre_allocate_memory(percentage=0.9):
   total_memory = torch.cuda.get_device_properties(0).total_memory
   memory_to_allocate = int(total_memory * percentage)
   num_floats = memory_to_allocate // 4  
   dummy_tensor = torch.empty(num_floats, dtype=torch.float32, device='cuda')
   return dummy_tensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--background_class", type=int, default=0)
    parser.add_argument("--ignore_label", type=int, default=255)

    # Introduce dynamic configuration loading
    parser.add_argument("--config", type=str, default=None, help="Path to experiment config file")
    args = parser.parse_args()

    # --- Dynamic configuration loading logic ---
    if args.config:
        import sys
        import importlib.util
        
        # 1. Load module from file path
        spec = importlib.util.spec_from_file_location("exp_config", args.config)
        exp_config_module = importlib.util.module_from_spec(spec) # type: ignore
        spec.loader.exec_module(exp_config_module) # type: ignore
        
        # 2. Get the cfg object
        if not hasattr(exp_config_module, 'cfg'):
            raise AttributeError(f"Config file {args.config} must define a 'cfg' object.")
        cfg = exp_config_module.cfg
        # print(f"[Info] Loaded configuration from: {args.config}")
    else:
        # Fall back to default configuration
        if default_cfg is None:
            raise ValueError("No config provided and default configs.config not found.")
        cfg = default_cfg
        print("[Info] Using default configuration from code_pascal_voc_2012/configs/config.py")
    # -----------------------

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high') # type: ignore
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids


    main(cfg, args)

    torch.cuda.empty_cache()