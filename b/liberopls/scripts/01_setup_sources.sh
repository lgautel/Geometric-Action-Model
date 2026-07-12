#!/usr/bin/env bash
# Clone DA3 / LIBERO / LIBERO-plus and download Plus assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

LOG="$LIBEROPLS_LOG_DIR/01_setup_sources.log"
exec > >(tee -a "$LOG") 2>&1

echo "[01] setup sources start $(date -Is)"
cd "$DA3_ROOT"

bash "$DA3_ROOT/scripts/setup_sources.sh"
bash "$DA3_ROOT/scripts/setup_libero_plus.sh" --download-assets

# Quick import checks (encoder path stubs addict; full DA3 api may need extra deps)
"$DA3_PYTHON" - <<'PY'
import sys
from pathlib import Path
root = Path(__import__("os").environ["DA3_ROOT"])
sys.path[:0] = [
    str(root / "src"),
    str(root / "LIBERO-plus"),
    str(root / "LIBERO"),
    str(root / "Depth-Anything-3" / "src"),
]
import libero
print("libero OK", getattr(libero, "__file__", None))
import robot.modeling.da3_giant_encoder as enc
print("da3_giant_encoder OK", enc.__file__)
PY

echo "[01] setup sources done $(date -Is)"
