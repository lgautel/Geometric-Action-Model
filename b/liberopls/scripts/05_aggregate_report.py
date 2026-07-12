#!/usr/bin/env python3
"""Aggregate LIBERO-Plus eval summaries and compare to paper claims."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PAPER = {
    "libero_plus_overall": 85.5,
    "camera": 83.1,
    "libero_orig": 97.6,
}

SUITE_KEYS = ("spatial", "object", "goal", "long")
CAT_ORDER = (
    "background",
    "camera",
    "language",
    "layout",
    "light",
    "noise",
    "robot",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_summaries(results_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for suite in SUITE_KEYS:
        suite_dir = results_root / suite
        if not suite_dir.is_dir():
            continue
        candidates = sorted(
            p
            for p in suite_dir.rglob("summary.json")
            if "shards" not in p.parts and "_archive" not in p.parts
        )
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            found[suite] = candidates[0]
    return found


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sr(summary: dict) -> float | None:
    for key in ("overall_success_rate", "average_success_rate", "success_rate", "sr"):
        if key in summary and summary[key] is not None:
            val = float(summary[key])
            return val * 100.0 if val <= 1.0 else val
    succ = (
        summary.get("total_successes")
        or summary.get("total_success")
        or summary.get("num_success")
    )
    trials = (
        summary.get("total_episodes")
        or summary.get("total_trials")
        or summary.get("num_trials")
    )
    if succ is not None and trials:
        return 100.0 * float(succ) / float(trials)
    return None


def _counts(summary: dict) -> tuple[int | None, int | None]:
    succ = summary.get("total_successes")
    trials = summary.get("total_episodes")
    if succ is None:
        succ = summary.get("total_success") or summary.get("num_success")
    if trials is None:
        trials = summary.get("total_trials") or summary.get("num_trials")
    if succ is None or trials is None:
        return None, None
    return int(succ), int(trials)


def _category_block(summary: dict) -> dict:
    block = summary.get("plus_official_category_results")
    if isinstance(block, dict) and block:
        return block
    for container_key in (
        "by_official_category",
        "official_category",
        "categories",
        "plus_categories",
    ):
        block = summary.get(container_key)
        if isinstance(block, dict) and block:
            return block
    return {}


def _category_srs(summary: dict) -> dict[str, dict]:
    """Return {cat: {sr_pct, num_success, num_trials}}."""
    out: dict[str, dict] = {}
    for name, val in _category_block(summary).items():
        key = str(name).lower()
        if isinstance(val, dict):
            sr = _sr(val)
            ns = val.get("num_success")
            nt = val.get("num_trials")
            out[key] = {
                "sr": sr,
                "num_success": int(ns) if ns is not None else None,
                "num_trials": int(nt) if nt is not None else None,
            }
        elif isinstance(val, (int, float)):
            sr = float(val) * 100.0 if float(val) <= 1.0 else float(val)
            out[key] = {"sr": sr, "num_success": None, "num_trials": None}
    return out


def main() -> int:
    root = _repo_root()
    results_root = Path(os.environ.get("OUT_ROOT", str(root / "b/liberopls/results")))
    report_path = results_root / "PAPER_COMPARE.md"
    status_path = results_root / "STATUS.md"
    summaries = _find_summaries(results_root)

    lines: list[str] = []
    lines.append("# LIBERO-Plus Reproduction vs Paper")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Results root: `{results_root}`")
    lines.append("- Protocol: official HF suite ckpts, `--plus`, `qpos=original`, 1 trial/task")
    lines.append("")
    lines.append("## Paper claims (GAM 1.4B)")
    lines.append("")
    lines.append("| Metric | Paper |")
    lines.append("|--------|------:|")
    lines.append(f"| LIBERO Orig. | {PAPER['libero_orig']} |")
    lines.append(f"| LIBERO-Plus overall | {PAPER['libero_plus_overall']} |")
    lines.append(f"| Camera split | {PAPER['camera']} |")
    lines.append("")

    missing = [s for s in SUITE_KEYS if s not in summaries]
    if missing:
        lines.append(f"**Incomplete suites:** `{', '.join(missing)}` — overall not final.")
        lines.append("")

    if not summaries:
        lines.append("## Local results")
        lines.append("")
        lines.append("_No suite-level `summary.json` found under results yet._")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n")
        status_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {report_path} (empty)")
        return 1

    lines.append("## Per-suite local results")
    lines.append("")
    lines.append("| Suite | Successes | Episodes | SR (%) | Elapsed (h) | summary |")
    lines.append("|-------|----------:|---------:|-------:|------------:|---------|")

    total_succ = 0
    total_trials = 0
    suite_srs: dict[str, float] = {}
    # Weighted category accumulation across suites
    cat_succ: dict[str, int] = {}
    cat_trials: dict[str, int] = {}
    per_suite_detail: list[str] = []

    for suite in SUITE_KEYS:
        path = summaries.get(suite)
        if path is None:
            lines.append(f"| {suite} | — | — | — | — | _missing_ |")
            continue
        data = _load(path)
        sr = _sr(data)
        succ, trials = _counts(data)
        if sr is not None:
            suite_srs[suite] = sr
        if succ is not None and trials is not None:
            total_succ += succ
            total_trials += trials
        elapsed = data.get("elapsed_sec")
        elapsed_h = f"{float(elapsed) / 3600.0:.2f}" if elapsed is not None else "n/a"
        sr_str = f"{sr:.2f}" if sr is not None else "n/a"
        succ_str = str(succ) if succ is not None else "n/a"
        trials_str = str(trials) if trials is not None else "n/a"
        rel = path.relative_to(results_root) if path.is_relative_to(results_root) else path
        lines.append(
            f"| {suite} | {succ_str} | {trials_str} | {sr_str} | {elapsed_h} | `{rel}` |"
        )

        cats = _category_srs(data)
        if cats:
            per_suite_detail.append(f"### {suite}")
            per_suite_detail.append("")
            per_suite_detail.append("| Category | Successes | Trials | SR (%) |")
            per_suite_detail.append("|----------|----------:|-------:|-------:|")
            for cat in CAT_ORDER:
                info = cats.get(cat)
                if not info:
                    continue
                ns, nt, csr = info["num_success"], info["num_trials"], info["sr"]
                if ns is not None and nt is not None:
                    cat_succ[cat] = cat_succ.get(cat, 0) + ns
                    cat_trials[cat] = cat_trials.get(cat, 0) + nt
                csr_s = f"{csr:.2f}" if csr is not None else "n/a"
                per_suite_detail.append(
                    f"| {cat} | {ns if ns is not None else 'n/a'} | "
                    f"{nt if nt is not None else 'n/a'} | {csr_s} |"
                )
            per_suite_detail.append("")

    overall = 100.0 * total_succ / total_trials if total_trials > 0 else None

    lines.append("")
    lines.append("## Aggregate vs paper")
    lines.append("")
    lines.append("| Metric | Paper | Local | Delta (pp) |")
    lines.append("|--------|------:|------:|-----------:|")
    if overall is not None:
        delta = overall - PAPER["libero_plus_overall"]
        lines.append(
            f"| LIBERO-Plus overall | {PAPER['libero_plus_overall']:.1f} | "
            f"{overall:.2f} | {delta:+.2f} |"
        )
        lines.append(
            f"| Episodes counted | — | {total_succ}/{total_trials} | — |"
        )
    else:
        lines.append(
            f"| LIBERO-Plus overall | {PAPER['libero_plus_overall']:.1f} | n/a | n/a |"
        )

    if cat_trials.get("camera"):
        cam = 100.0 * cat_succ["camera"] / cat_trials["camera"]
        lines.append(
            f"| Camera split | {PAPER['camera']:.1f} | {cam:.2f} | "
            f"{cam - PAPER['camera']:+.2f} |"
        )
    else:
        lines.append(
            f"| Camera split | {PAPER['camera']:.1f} | n/a | n/a |"
        )

    lines.append("")
    lines.append("## Official Plus categories (weighted across finished suites)")
    lines.append("")
    lines.append("| Category | Successes | Trials | SR (%) |")
    lines.append("|----------|----------:|-------:|-------:|")
    for cat in CAT_ORDER:
        if cat not in cat_trials:
            continue
        sr_c = 100.0 * cat_succ[cat] / cat_trials[cat]
        lines.append(
            f"| {cat} | {cat_succ[cat]} | {cat_trials[cat]} | {sr_c:.2f} |"
        )

    lines.append("")
    lines.append("## Per-suite category breakdown")
    lines.append("")
    lines.extend(per_suite_detail)

    lines.append("## Verdict")
    lines.append("")
    if missing:
        lines.append(
            f"Incomplete: waiting on `{', '.join(missing)}`. "
            "Do not treat overall as paper-comparable until all four suites finish."
        )
    elif overall is not None:
        if abs(overall - PAPER["libero_plus_overall"]) <= 2.0:
            lines.append(
                f"Local overall **{overall:.2f}%** is within 2pp of paper "
                f"**{PAPER['libero_plus_overall']}%**. Reproduction looks consistent."
            )
        else:
            lines.append(
                f"Local overall **{overall:.2f}%** differs from paper "
                f"**{PAPER['libero_plus_overall']}%** by >2pp. Check qpos=original, "
                "prompt normalization, checkpoints, and Plus assets."
            )
        if cat_trials.get("camera"):
            cam = 100.0 * cat_succ["camera"] / cat_trials["camera"]
            lines.append(
                f"Camera: local **{cam:.2f}%** vs paper **{PAPER['camera']}%** "
                f"({cam - PAPER['camera']:+.2f} pp)."
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- First all-run failed on `flex_attention` compile batch-dim mismatch; "
        "fixed by eager flex + `MAX_BATCH_SIZE=1`."
    )
    lines.append(
        "- First goal run aborted on env-worker init timeout (900s); resumed with "
        "`ENV_WORKER_TIMEOUT_SEC=1800` via `scripts/07_resume_goal_long.sh`."
    )

    text = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)
    status_path.write_text(text)
    print(f"Wrote {report_path}")
    print(f"Wrote {status_path}")
    if overall is not None:
        print(
            f"Overall Plus SR: {overall:.2f}% "
            f"({total_succ}/{total_trials}; paper {PAPER['libero_plus_overall']}%)"
            + (f"; missing={missing}" if missing else "")
        )
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
