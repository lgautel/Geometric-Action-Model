#!/usr/bin/env python3
"""Draw freeze vs trainable modules and gradient flow for GAM training."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = Path(__file__).with_name("freeze_gradient_map.png")

FROZEN = "#BDD7EE"
TRAINABLE = "#C6EFCE"
TEACHER = "#D9D9D9"


def box(ax, xy, w, h, text, fc, fontsize=9):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2,
            facecolor=fc,
            edgecolor="#333333",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, a, b, color="#C00000", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=12, linewidth=1.5, color=color
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(
        "Training: frozen modules vs. gradient-updated modules",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Teacher column
    box(ax, (0.4, 3.8), 2.4, 1.6, "Teacher DA3\n(all blocks)\nrequires_grad=False", TEACHER)
    box(ax, (0.4, 2.0), 2.4, 1.2, "Future visual\ntargets $Z^{(L_s)}$\n(no grad)", TEACHER, 8)

    # Student shallow
    box(ax, (3.4, 4.2), 2.6, 1.2, "Student DA3\nblocks 0–12\nFROZEN", FROZEN)
    box(ax, (3.4, 2.6), 2.6, 1.0, "DPT / cam_dec\nFROZEN", FROZEN, 8)
    box(ax, (3.4, 1.2), 2.6, 1.0, "CLIP/T5 encoder\nFROZEN", FROZEN, 8)

    # Trainable
    box(ax, (6.6, 4.4), 2.6, 1.2, "GAMFuturePredictor\n12L  TRAINABLE", TRAINABLE)
    box(ax, (6.6, 2.8), 2.6, 1.2, "Student DA3\nblocks 13–39\nTRAINABLE", TRAINABLE)
    box(ax, (6.6, 1.2), 2.6, 1.2, "ActionHeadV2\n(+ action tokens)\nTRAINABLE", TRAINABLE)

    # Losses
    box(ax, (9.8, 4.0), 1.8, 1.0, "$\\mathcal{L}_{feat}$", "#F8CBAD", 10)
    box(ax, (9.8, 2.6), 1.8, 1.0, "$\\mathcal{L}_{act}$", "#F8CBAD", 10)
    box(ax, (9.8, 1.2), 1.8, 1.0, "$\\mathcal{L}_{depth}$", "#F8CBAD", 10)

    # Forward dashed (gray)
    arrow(ax, (2.8, 4.6), (3.4, 4.8), color="#666666")
    arrow(ax, (6.0, 4.8), (6.6, 5.0), color="#666666")
    arrow(ax, (9.2, 5.0), (9.8, 4.5), color="#666666")
    arrow(ax, (9.2, 3.4), (9.8, 3.1), color="#666666")
    arrow(ax, (9.2, 1.8), (9.8, 1.7), color="#666666")
    arrow(ax, (2.8, 2.6), (9.8, 4.3), color="#888888", style="->")

    # Backward red
    arrow(ax, (9.8, 4.5), (9.2, 5.0), color="#C00000")
    arrow(ax, (6.6, 5.0), (6.0, 4.8), color="#C00000")
    # stop before frozen shallow: X mark via text
    ax.text(6.2, 4.55, "∇ stops", color="#C00000", fontsize=8, ha="center")

    ax.legend(
        handles=[
            Patch(facecolor=FROZEN, edgecolor="#333", label="Frozen (no grad)"),
            Patch(facecolor=TRAINABLE, edgecolor="#333", label="Trainable"),
            Patch(facecolor=TEACHER, edgecolor="#333", label="Teacher (target only)"),
            Patch(facecolor="#F8CBAD", edgecolor="#333", label="Loss terms"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    ax.text(
        6.0,
        0.35,
        "Config: da3_finetune.freeze_blocks_before=13  ↔  paper split layer $L_s=12$",
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
