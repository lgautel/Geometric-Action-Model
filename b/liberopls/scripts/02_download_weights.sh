#!/usr/bin/env bash
# Download released GAM suite checkpoints and DA3 base weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/02_download_weights.log"
exec > >(tee -a "$LOG") 2>&1

echo "[02] download weights start $(date -Is)"
mkdir -p "$DA3_ROOT/checkpoints" "$HF_ROOT"

echo "[02] hf download SeonghuJeon/3da-libero-gam -> $HF_ROOT"
hf download SeonghuJeon/3da-libero-gam --local-dir "$HF_ROOT"

BASE_CKPT="$DA3_BASE_CKPT"
if [[ -f "$BASE_CKPT" ]]; then
  echo "[02] DA3 base ckpt already present: $BASE_CKPT"
else
  echo "[02] downloading track4world_da3.pth from TencentARC/Track4World (~5.5GB)"
  mkdir -p "$(dirname "$BASE_CKPT")"
  hf download TencentARC/Track4World track4world_da3.pth \
    --local-dir "$DA3_ROOT/checkpoints"
  if [[ ! -f "$BASE_CKPT" ]]; then
    # hf may nest under a subdir; locate and link
    FOUND="$(find "$DA3_ROOT/checkpoints" -name 'track4world_da3.pth' | head -1)"
    if [[ -n "$FOUND" && "$FOUND" != "$BASE_CKPT" ]]; then
      ln -sfn "$FOUND" "$BASE_CKPT"
    fi
  fi
fi
test -f "$BASE_CKPT" || { echo "FATAL: missing $BASE_CKPT"; exit 1; }

echo "[02] verifying suite files"
for suite in spatial object goal long; do
  test -f "$HF_ROOT/$suite/gam.pt" || { echo "MISSING $HF_ROOT/$suite/gam.pt"; exit 1; }
  test -f "$HF_ROOT/$suite/config.yaml" || { echo "MISSING $HF_ROOT/$suite/config.yaml"; exit 1; }
  echo "  OK $suite ($(du -h "$HF_ROOT/$suite/gam.pt" | awk '{print $1}'))"
done

ls -lh "$BASE_CKPT"
echo "[02] download weights done $(date -Is)"
