#!/usr/bin/env python3
"""Illustrate per-timestep token layout inside GAMFuturePredictor."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("token_layout.png")


def tok(ax, x, y, w, h, label, color, fs=8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=1.0,
            facecolor=color,
            edgecolor="#333333",
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 4.6), dpi=160)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title(
        "GAMFuturePredictor token layout (one context window)",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )

    # Language
    tok(ax, 0.3, 3.2, 2.0, 0.9, "Language\nCLIP 77×768\ncross-attn", "#E2D5F1", 8)

    # Timestep blocks
    colors = ["#D9EAD3", "#FFF2CC", "#FCE5CD"]
    titles = ["$t{-}H{+}1$", r"$\cdots$", "$t$ (current)"]
    xs = [2.8, 6.0, 8.0]
    for i, (x, c, title) in enumerate(zip(xs, colors, titles)):
        ax.text(x + 1.4, 4.15, title, ha="center", fontsize=9, fontweight="bold")
        # view tokens
        tok(ax, x, 3.2, 1.5, 0.7, "view0\nCLS+R+P", c, 7)
        tok(ax, x + 1.55, 3.2, 1.5, 0.7, "view1\nCLS+R+P", c, 7)
        tok(ax, x, 2.3, 1.5, 0.65, "proprio $s$", "#CFE2F3", 7)
        tok(ax, x + 1.55, 2.3, 1.5, 0.65, "prev act $a$", "#F4CCCC", 7)

    # Block-causal note
    ax.annotate(
        "",
        xy=(11.0, 2.9),
        xytext=(2.8, 2.9),
        arrowprops=dict(arrowstyle="<->", color="#666"),
    )
    ax.text(6.9, 1.85, "block-causal self-attention (no future leakage)", ha="center", fontsize=9)

    # Outputs
    tok(ax, 2.8, 0.55, 2.8, 0.9, "predict $\\tilde Z_{t'+1}^{(L_s)}$\n(next visual tokens)", "#D9EAD3", 8)
    tok(ax, 6.0, 0.55, 2.8, 0.9, "predict $\\tilde a_{t'}$\n→ replicate ×V views", "#F4CCCC", 8)
    tok(ax, 9.2, 0.55, 2.8, 0.9, "optional $\\tilde s_{t'+1}$\nproprio future", "#CFE2F3", 8)

    ax.text(
        6.25,
        0.15,
        "Code: past_visual [B,H,V,1+R+P,1536] + proprio + past_action → GAMFuturePredictor.forward",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
