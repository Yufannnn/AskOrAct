"""Explainer figure for what K (ambiguity) means in the Ask-or-Act setup.
Three panels, same instruction, K in {2, 4, 6} candidate goals."""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_PATH = "results/figures/paper/k_explainer.png"

PANELS = [2, 4, 6]
INSTRUCTION = '"bring the gem"'
GEM_COLOR = "#c0392b"   # matches AskRed in slides
BUBBLE_FC = "#f4f4f4"
BUBBLE_EC = "#7a7a7a"


def draw_speech_bubble(ax, x, y, text):
    bbox = dict(boxstyle="round,pad=0.45,rounding_size=0.30",
                fc=BUBBLE_FC, ec=BUBBLE_EC, lw=1.1)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=11, style="italic", bbox=bbox, zorder=3)
    # little tail under the bubble
    ax.plot([x - 0.08, x + 0.02, x - 0.18],
            [y - 0.22, y - 0.22, y - 0.42],
            color=BUBBLE_EC, lw=1.1, zorder=3)


def draw_principal(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.17, color="#222222", zorder=4))
    ax.text(x, y, "P", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5)


def gem_positions(k):
    # spread the K gems on a gentle arc so they all look equally "candidate"
    if k == 1:
        return [(1.2, 0.0)]
    xs = np.linspace(-0.9, 0.9, k)
    ys = -0.35 + 0.25 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, k))
    return list(zip(xs * 1.15 + 0.7, ys))


def draw_gem(ax, x, y):
    # diamond "gem" shape
    poly = np.array([
        [x, y + 0.18],
        [x + 0.14, y],
        [x, y - 0.18],
        [x - 0.14, y],
    ])
    ax.add_patch(mpatches.Polygon(poly, closed=True,
                                   facecolor=GEM_COLOR, edgecolor="#6e1a12",
                                   lw=0.9, zorder=4))
    # little highlight
    ax.plot([x - 0.05], [y + 0.05], marker=".", color="white",
            markersize=3, zorder=5)


def draw_panel(ax, k):
    ax.set_xlim(-1.5, 2.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # background
    ax.add_patch(mpatches.FancyBboxPatch(
        (-1.45, -1.25), 3.75, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc="#ffffff", ec="#cfcfcf", lw=0.9))

    # principal + bubble
    draw_principal(ax, -1.0, -0.35)
    draw_speech_bubble(ax, -0.55, 0.55, INSTRUCTION)

    # candidate gems
    for (gx, gy) in gem_positions(k):
        draw_gem(ax, gx, gy)

    # panel title (K value)
    ax.text(0.4, 1.05, rf"$K={k}$", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#173f5f")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    for ax, k in zip(axes, PANELS):
        draw_panel(ax, k)

    fig.suptitle(
        r"Ambiguity $K$ = number of candidate goals that fit the same instruction",
        fontsize=11, y=1.00, color="#333333",
    )

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT_PATH)


if __name__ == "__main__":
    main()
