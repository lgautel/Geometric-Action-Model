#!/usr/bin/env bash
# Activate GAM venv and export paths for LIBERO-Plus reproduction.
# Usage (bash): source b/liberopls/env.sh
# Prefer: bash -c 'source b/liberopls/env.sh && ...'

# Resolve this file's directory under bash or zsh.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _LIBEROPLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # shellcheck disable=SC2296
  _LIBEROPLS_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  _LIBEROPLS_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
_REPO_ROOT="$(cd "$_LIBEROPLS_DIR/../.." && pwd)"

# Always pin to this checkout unless LIBEROPLS_KEEP_DA3_ROOT=1.
if [[ "${LIBEROPLS_KEEP_DA3_ROOT:-0}" != "1" ]]; then
  export DA3_ROOT="$_REPO_ROOT"
else
  export DA3_ROOT="${DA3_ROOT:-$_REPO_ROOT}"
fi
export DA3_CODE_ROOT="$DA3_ROOT"
export VENV_GAM="${VENV_GAM:-/mnt/r/VENV/gam}"
export DA3_PYTHON="${DA3_PYTHON:-$VENV_GAM/bin/python}"
export PATH="$VENV_GAM/bin:$PATH"

export DA3_LIBERO_SOURCE_DIR="$DA3_ROOT/LIBERO"
export DA3_LIBERO_PLUS_DIR="$DA3_ROOT/LIBERO-plus"
export PYTHONPATH="$DA3_ROOT/src:$DA3_LIBERO_PLUS_DIR:$DA3_LIBERO_SOURCE_DIR:$DA3_ROOT/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-device}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Always re-derive weight/result paths from DA3_ROOT (ignore stale exports).
export DA3_BASE_CKPT="$DA3_ROOT/checkpoints/track4world_da3.pth"
export GAM_PRETRAINED_CKPT="$DA3_ROOT/checkpoints_hf/3da-libero-gam/pretrained/pretrained-gam.pt"
export HF_ROOT="$DA3_ROOT/checkpoints_hf/3da-libero-gam"
export OUT_ROOT="$DA3_ROOT/b/liberopls/results"
export LIBEROPLS_LOG_DIR="$DA3_ROOT/b/liberopls/logs"

mkdir -p "$OUT_ROOT" "$LIBEROPLS_LOG_DIR"

if [[ -f "$VENV_GAM/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_GAM/bin/activate"
fi

echo "[liberopls/env] DA3_ROOT=$DA3_ROOT"
echo "[liberopls/env] DA3_PYTHON=$DA3_PYTHON"
echo "[liberopls/env] HF_ROOT=$HF_ROOT"
echo "[liberopls/env] OUT_ROOT=$OUT_ROOT"
