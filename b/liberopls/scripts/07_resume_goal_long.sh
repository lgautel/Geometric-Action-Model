#!/usr/bin/env bash
# Resume remaining LIBERO-Plus suites after a partial all-run.
# Completes: goal (re-run fresh) + long. spatial/object already finished.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/07_resume_goal_long.log"
exec > >(tee -a "$LOG") 2>&1

echo "[07] resume goal+long start $(date -Is)"

export GAM_EVAL_GPUS="${GAM_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
export OUT_ROOT
export HF_ROOT
export DA3_BASE_CKPT
export DA3_PYTHON
export PARALLEL_ENVS_PER_GPU="${PARALLEL_ENVS_PER_GPU:-8}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.05}"
export ENV_WORKER_TIMEOUT_SEC="${ENV_WORKER_TIMEOUT_SEC:-1800}"
export ROLLOUT_WALL_TIMEOUT_SEC="${ROLLOUT_WALL_TIMEOUT_SEC:-1800}"
export DA3_COMPILE_INFERENCE="${DA3_COMPILE_INFERENCE:-none}"

# Archive incomplete goal run if present (no suite-level summary).
if [[ -d "$OUT_ROOT/goal" ]]; then
  if [[ ! -f "$(find "$OUT_ROOT/goal" -maxdepth 2 -name summary.json ! -path '*/shards/*' | head -1)" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "$OUT_ROOT/_archive"
    mv "$OUT_ROOT/goal" "$OUT_ROOT/_archive/goal_incomplete_$ts"
    echo "[07] archived incomplete goal -> _archive/goal_incomplete_$ts"
  fi
fi

echo "[07] running goal (timeout=${ENV_WORKER_TIMEOUT_SEC}s, batch=${MAX_BATCH_SIZE})"
bash "$DA3_ROOT/scripts/run_hf_gam_libero_plus_eval.sh" goal

echo "[07] running long"
bash "$DA3_ROOT/scripts/run_hf_gam_libero_plus_eval.sh" long

echo "[07] resume done $(date -Is)"
echo "[07] next: python b/liberopls/scripts/05_aggregate_report.py"
