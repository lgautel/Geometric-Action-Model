#!/usr/bin/env python3
"""Bar charts for GAM component and split-layer ablations (paper tables)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).with_name("ablation_bars.png")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    # --- Left: component ablation (Object suite numbers from paper) ---
    ax = axes[0]
    labels = [
        "Full\n(H=1)",
        "H=2",
        "H=4",
        "w/o depth",
        "w/o depth\n& feat",
        "w/o feat",
        "no pretrain",
        "no PT\nw/o depth",
        "no PT\nw/o feat",
        "no PT\nw/o both",
    ]
    orig = [99.6, 97.2, 98.2, 98.4, 98.6, 99.6, 98.4, 95.2, 96.4, 93.6]
    plus = [89.7, 84.4, 85.1, 89.0, 89.5, 89.7, 73.4, 66.5, 80.0, 50.0]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, orig, w, label="LIBERO Orig.", color="#4C78A8")
    ax.bar(x + w / 2, plus, w, label="LIBERO-Plus", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Component ablation (LIBERO-Object)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.axhline(89.7, color="#F58518", ls="--", lw=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right: split layer ---
    ax = axes[1]
    ls = [0, 12, 19, 27, 33, 39]
    orig_ls = [5.4, 99.6, 95.6, 1.2, 0.0, 0.0]
    plus_ls = [1.8, 70.1, 63.4, 1.6, 0.0, 0.0]
    x = np.arange(len(ls))
    ax.bar(x - w / 2, orig_ls, w, label="LIBERO Orig.", color="#4C78A8")
    ax.bar(x + w / 2, plus_ls, w, label="LIBERO-Plus", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$L_s$={v}" for v in ls], fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Split-layer ablation (no $L_{depth}$)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate(
        "sweet spot:\nbefore DPT\n& after shallow",
        xy=(1, 99.6),
        xytext=(2.2, 75),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#333"),
        color="#333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Ablation evidence from GAM paper (Object suite / layer study)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
