#!/usr/bin/env bash
# Smoke checks before full LIBERO-Plus eval.
# SMOKE_MODE=quick (default): imports + ckpt/config presence + CUDA
# SMOKE_MODE=spatial: run official spatial suite on GAM_EVAL_GPUS (slow)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/03_smoke_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "[03] smoke eval start $(date -Is)"
SMOKE_MODE="${SMOKE_MODE:-quick}"

test -x "$DA3_PYTHON" || { echo "missing DA3_PYTHON=$DA3_PYTHON"; exit 1; }
test -f "$DA3_BASE_CKPT" || { echo "missing DA3_BASE_CKPT=$DA3_BASE_CKPT"; exit 1; }
test -d "$DA3_LIBERO_PLUS_DIR/libero" || { echo "missing LIBERO-plus"; exit 1; }
test -d "$DA3_LIBERO_PLUS_DIR/libero/libero/assets" || { echo "missing Plus assets"; exit 1; }

for suite in spatial object goal long; do
  test -f "$HF_ROOT/$suite/gam.pt" || { echo "missing $suite/gam.pt"; exit 1; }
  test -f "$HF_ROOT/$suite/config.yaml" || { echo "missing $suite/config.yaml"; exit 1; }
done

"$DA3_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
root = Path(os.environ["DA3_ROOT"])
sys.path[:0] = [str(root / "src"), str(root / "LIBERO-plus"), str(root / "LIBERO"), str(root / "Depth-Anything-3" / "src")]
import torch
print("torch", torch.__version__, "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1
import robot.modeling.da3_giant_encoder as enc
print("da3_giant_encoder OK", enc.__file__)
import libero
print("libero OK")
# Touch HF config
from omegaconf import OmegaConf
cfg = OmegaConf.load(root / "checkpoints_hf/3da-libero-gam/spatial/config.yaml")
print("spatial config keys", list(cfg.keys())[:8])
print("SMOKE_QUICK_OK")
PY

if [[ "$SMOKE_MODE" == "spatial" ]]; then
  export GAM_EVAL_GPUS="${GAM_EVAL_GPUS:-0}"
  export PARALLEL_ENVS_PER_GPU="${PARALLEL_ENVS_PER_GPU:-4}"
  export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
  echo "[03] running spatial suite on GPUs=$GAM_EVAL_GPUS"
  bash "$DA3_ROOT/scripts/run_hf_gam_libero_plus_eval.sh" spatial
fi

echo "[03] smoke eval done $(date -Is)"
