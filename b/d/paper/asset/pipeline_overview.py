#!/usr/bin/env python3
"""Draw GAM three-stage pipeline overview for the paper note."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).with_name("pipeline_overview.png")


def box(ax, xy, w, h, text, fc, ec="#333333", fontsize=10, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1a1a1a",
        wrap=True,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color="#444444",
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=160)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title(
        "GAM Pipeline: one geometric backbone for perception, prediction, and action",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Inputs
    box(ax, (0.3, 3.5), 1.8, 1.2, "Multi-view RGB\n+ proprio\n+ language", "#E8F1F8")
    box(ax, (0.3, 1.6), 1.8, 1.0, "Action history\n$a_{t-H:t-1}$", "#F5F0E6")

    # Stage 1
    box(
        ax,
        (2.6, 2.0),
        2.4,
        2.4,
        "① Observation Encoder\nDA3 blocks 0–12\n(frozen)\n\n$Z^{(L_s)}$\nencode_shallow_\nvisual_slots",
        "#D9EAD3",
        weight="bold",
        fontsize=9,
    )

    # Stage 2
    box(
        ax,
        (5.5, 2.0),
        2.6,
        2.4,
        "② Causal Future Predictor\n12-layer transformer\n(trainable)\n\n$\\tilde Z_{t+1},\\;\\tilde a_t$\nGAMFuturePredictor",
        "#FFF2CC",
        weight="bold",
        fontsize=9,
    )

    # Stage 3
    box(
        ax,
        (8.6, 2.0),
        2.6,
        2.4,
        "③ Deep Propagation\nDA3 blocks 13–39\n+ ActionHeadV2\n\nfuture depth +\n7-DoF action chunk",
        "#FCE5CD",
        weight="bold",
        fontsize=9,
    )

    # Outputs
    box(ax, (11.4, 3.4), 0.9, 0.9, "Action\nchunk", "#EAD1DC", fontsize=9)
    box(ax, (11.4, 2.0), 0.9, 0.9, "Future\ndepth", "#D0E2F3", fontsize=9)

    arrow(ax, (2.1, 4.1), (2.6, 3.5))
    arrow(ax, (2.1, 2.1), (2.6, 2.8))
    arrow(ax, (5.0, 3.2), (5.5, 3.2))
    arrow(ax, (8.1, 3.2), (8.6, 3.2))
    arrow(ax, (11.2, 3.6), (11.4, 3.85))
    arrow(ax, (11.2, 2.8), (11.4, 2.45))

    ax.text(
        6.25,
        0.55,
        "Split at $L_s=12$  ·  Teacher DA3 provides frozen future-feature targets during training",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
