#!/bin/bash
set -e
set -x

# Get the absolute path of the directory containing this script
EXP_DIR=$(cd "$(dirname "$0")"; pwd)
# Project root directory (relative to Exp00... is two levels up)
CODE_ROOT=$(cd "$EXP_DIR/../.."; pwd)

# Switch to VOC code root, since many relative-path imports in train_voc.py depend on this
cd "$CODE_ROOT"

echo "Running experiment from: $EXP_DIR"
echo "Codebase: $(pwd)"

# Set environment variable (preserve original logic)
export TOKENIZERS_PARALLELISM=false

# Launch training, passing the config.py from the current experiment directory
# train_coco.py already supports the --config argument
python train_coco.py --config "$EXP_DIR/config.py"