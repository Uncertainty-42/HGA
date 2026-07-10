#!/bin/bash
set -e

# capture the signal to exit, cleaning up the CRF background process 
cleanup() {
    if [ -n "${CRF_PID:-}" ] && kill -0 "$CRF_PID" 2>/dev/null; then
        kill "$CRF_PID" 2>/dev/null
        wait "$CRF_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# ============================================================
# val_voc.bash - Co2SAM single-scaled validation/test (VOC 2012)
# usage: bash val_voc.bash
# just modify the params below. no need to adjust the command
# ============================================================

# ---- required params ----
MODEL_PATH=""
INFER_SET="val"          # val or test

# ---- optional params ----
GPU="6"                  # GPU ID
# generated new dir automatically if blank
OUTPUT_ROOT=""


CRF="yes"                # enter "yes" to enable CRF post-processing, leave blank to disable it
VIZ="yes"                   # enter "yes" to enable visualization, leave blank to disable it
# available validating probe:
#   baseline_monitoring, top2_edge_alignment_monitoring, dino_boxes,
#   val_alignment_check, visualization_error_analysis
VIZ_PROBES="baseline_monitoring,dino_boxes,val_alignment_check,visualization_error_analysis"
# ---- visualizer policy controll ----
VIZ_WARMUP_STEPS=0
VIZ_WARMUP_FREQ=1
VIZ_EPOCH1_FREQ=10
VIZ_LATER_FREQ=10


# ---- construct the command automatically ----
ARGS=(--model_path "$MODEL_PATH" --infer_set "$INFER_SET" --gpu "$GPU")


if [ -n "$OUTPUT_ROOT" ]; then
    ARGS+=(--output_root "$OUTPUT_ROOT")
fi

if [ -n "$CRF" ]; then
    ARGS+=(--crf_post)
fi
if [ -n "$VIZ" ]; then
    ARGS+=(--viz_eval_enabled)
    if [ -n "$VIZ_PROBES" ]; then
        ARGS+=(--viz_probes "$VIZ_PROBES")
    fi
    ARGS+=(--viz_warmup_steps "$VIZ_WARMUP_STEPS")
    ARGS+=(--viz_warmup_freq "$VIZ_WARMUP_FREQ")
    ARGS+=(--viz_epoch1_freq "$VIZ_EPOCH1_FREQ")
    ARGS+=(--viz_later_freq "$VIZ_LATER_FREQ")

fi
if [ -n "$NAMLAB_DIR" ]; then
    ARGS+=(--namlab_dir "$NAMLAB_DIR")
fi
if [ -n "$DEPTH_DIR" ]; then
    ARGS+=(--depth_dir "$DEPTH_DIR")
fi

# switch to the directory of the script (root directory of the project)
cd "$(dirname "$0")"


if [ -n "$CRF" ]; then
    CRF_PID=""
    IMAGES_DIR="path/to/your/VOC2012/JPEGImages"
    GT_DIR="path/to/your/VOC2012/SegmentationClassAug"
    N_CLASS=21
    N_JOBS=20

    CRF_ARGS=(--model_path "$MODEL_PATH" --infer_set "$INFER_SET")
    CRF_ARGS+=(--images_dir "$IMAGES_DIR")
    CRF_ARGS+=(--gt_dir "$GT_DIR")
    CRF_ARGS+=(--n_class "$N_CLASS")
    CRF_ARGS+=(--n_jobs "$N_JOBS")
    if [ -n "$OUTPUT_ROOT" ]; then
        CRF_ARGS+=(--output_root "$OUTPUT_ROOT")
    fi

    conda run -n hga_crf_env python crf_only.py "${CRF_ARGS[@]}" & CRF_PID=$!
fi

echo "=========================================="
echo "validating: python val_voc.py ${ARGS[@]}"
echo "=========================================="

python val_voc.py "${ARGS[@]}"

if [ -n "$CRF" ] && [ -n "$CRF_PID" ]; then
    wait $CRF_PID
fi