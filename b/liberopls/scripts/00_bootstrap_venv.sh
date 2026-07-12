#!/usr/bin/env bash
# Create /mnt/r/VENV/gam, install torch + requirements, editable-install this repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/00_bootstrap_venv.log"
exec > >(tee -a "$LOG") 2>&1

echo "[00] bootstrap start $(date -Is)"
echo "[00] VENV_GAM=$VENV_GAM"
echo "[00] DA3_ROOT=$DA3_ROOT"

if [[ ! -x "$VENV_GAM/bin/python" ]]; then
  echo "[00] creating venv at $VENV_GAM"
  python3 -m venv "$VENV_GAM"
fi

# shellcheck disable=SC1091
source "$VENV_GAM/bin/activate"
python -m pip install -U pip setuptools wheel

echo "[00] installing torch 2.5.1 + cu124"
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[00] installing requirements.txt"
pip install -r "$DA3_ROOT/requirements.txt"

echo "[00] editable install da3-gam-libero"
pip install -e "$DA3_ROOT"

echo "[00] force opencv-python-headless"
pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
pip install --no-deps "opencv-python-headless==4.11.0.86"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
import robot, gam
print("robot/gam import OK", robot.__file__ if hasattr(robot, "__file__") else robot, gam)
PY

echo "[00] bootstrap done $(date -Is)"
