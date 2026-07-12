#!/usr/bin/env bash
# Full LIBERO-Plus eval on 8×H200 for all suites (paper reproduction).
#
# IMPORTANT: MAX_BATCH_SIZE defaults to 1. The GAM predictor compiles
# flex_attention with torch.compile(dynamic=False); variable batch sizes from
# the batched harness trigger BackendCompilerFailed ("Batch dimension must
# match") and collapse success rate. Env-level parallelism still comes from
# PARALLEL_ENVS_PER_GPU / sharding across GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/04_run_libero_plus_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "[04] full LIBERO-Plus eval start $(date -Is)"

export GAM_EVAL_GPUS="${GAM_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
export OUT_ROOT
export HF_ROOT
export DA3_BASE_CKPT
export DA3_PYTHON
export PARALLEL_ENVS_PER_GPU="${PARALLEL_ENVS_PER_GPU:-16}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.05}"
export GAM_PLUS_PERTURBATION="${GAM_PLUS_PERTURBATION:-all}"
export GAM_PLUS_OFFICIAL_CATEGORY="${GAM_PLUS_OFFICIAL_CATEGORY:-all}"
# Keep default inference compile off (CUDA-graph path is opt-in).
export DA3_COMPILE_INFERENCE="${DA3_COMPILE_INFERENCE:-none}"

echo "[04] GAM_EVAL_GPUS=$GAM_EVAL_GPUS"
echo "[04] OUT_ROOT=$OUT_ROOT"
echo "[04] PARALLEL_ENVS_PER_GPU=$PARALLEL_ENVS_PER_GPU MAX_BATCH_SIZE=$MAX_BATCH_SIZE"

bash "$DA3_ROOT/scripts/run_hf_gam_libero_plus_eval.sh" all

echo "[04] full eval done $(date -Is)"
echo "[04] next: python b/liberopls/scripts/05_aggregate_report.py"
