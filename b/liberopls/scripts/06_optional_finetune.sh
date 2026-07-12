#!/usr/bin/env bash
# Optional suite finetune (does NOT run by default). Requires LIBERO HDF5 data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/06_optional_finetune.log"
exec > >(tee -a "$LOG") 2>&1

echo "[06] optional finetune"
echo "Requires:"
echo "  - $GAM_PRETRAINED_CKPT"
echo "  - data/libero_noop/<suite>/*.hdf5 and _stats/"
echo "This script is a helper only; paper Plus numbers are reproduced via eval of HF ckpts."

if [[ "${CONFIRM_FINETUNE:-}" != "1" ]]; then
  echo "Set CONFIRM_FINETUNE=1 to actually launch DeepSpeed training. Exiting."
  exit 0
fi

test -f "$GAM_PRETRAINED_CKPT" || { echo "missing pretrained ckpt"; exit 1; }
test -d "$DA3_ROOT/data/libero_noop" || { echo "missing data/libero_noop"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$DA3_ROOT"
deepspeed --include localhost:0,1,2,3,4,5,6,7 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --set stage_1.ckpt_path="$GAM_PRETRAINED_CKPT" \
  --wandb-name gam_liberopls_finetune \
  "$@"
