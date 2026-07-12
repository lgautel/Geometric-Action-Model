#!/usr/bin/env bash
# Poll until goal+long suite summaries exist, then aggregate.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/b/liberopls/env.sh"
LOG="$LIBEROPLS_LOG_DIR/08_wait_and_aggregate.log"
exec >>"$LOG" 2>&1

need_summary() {
  local suite="$1"
  find "$OUT_ROOT/$suite" -maxdepth 2 -name summary.json ! -path '*/shards/*' 2>/dev/null | head -1 || true
}

progress_count() {
  local suite="$1"
  local n=0
  if [[ -d "$OUT_ROOT/$suite" ]]; then
    n=$(find "$OUT_ROOT/$suite" -name progress.jsonl -print0 2>/dev/null | xargs -0 cat 2>/dev/null | wc -l | tr -d ' ')
  fi
  echo "${n:-0}"
}

proc_alive() {
  local n
  n=$(pgrep -f 'run_hf_gam_libero_plus_eval|eval_libero_unified|07_resume_goal' 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

echo "[08] wait start $(date -Is)"
while true; do
  g=$(need_summary goal)
  l=$(need_summary long)
  gp=$(progress_count goal)
  lp=$(progress_count long)
  alive=$(proc_alive)
  echo "[08] $(date -Is) goal_prog=${gp} long_prog=${lp} goal_sum=${g:-none} long_sum=${l:-none} procs=${alive}"
  if [[ -n "${g}" && -n "${l}" ]]; then
    echo "[08] both summaries ready"
    break
  fi
  if [[ "${alive}" -eq 0 ]]; then
    echo "[08] ERROR: eval processes died before finishing" >&2
    python "$ROOT/b/liberopls/scripts/05_aggregate_report.py" || true
    exit 1
  fi
  sleep 120
done

python "$ROOT/b/liberopls/scripts/05_aggregate_report.py"
echo "[08] done $(date -Is)"
