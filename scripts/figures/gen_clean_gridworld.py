"""Clean two-room gridworld for embedding in an Excalidraw figure.

Produces results/figures/exploratory/gridworld_clean.png (cells + walls + doorway + dots + P/A
+ dashed gold ring on the true goal). NO side panel and NO "true goal"
callout — those are added on top in setup_overview.excalidraw.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

N = 9
MID = 4
DOOR_ROW = 4

# (row, col, fill, label)  ── label only used for principal/assistant
OBJECTS = [
    # left room ── distractors + agents
    (1, 1, "#3498db", None),  # blue
    (2, 1, "#222222", "P"),   # principal
    (3, 2, "#e74c3c", None),  # red distractor
    (5, 1, "#e67e22", "A"),   # assistant
    (6, 2, "#2ecc71", None),  # green
    # right room
    (1, 6, "#e74c3c", None),  # red distractor
    (5, 7, "#e74c3c", "TRUE"),  # true goal (gets dashed gold ring)
]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(-0.5, N - 0.5)

for r in range(N):
    for c in range(N):
        is_outer = (r == 0 or r == N - 1 or c == 0 or c == N - 1)
        is_mid_wall = (c == MID and r != DOOR_ROW)
        if is_outer or is_mid_wall:
            fc = "#555555"
        elif c < MID:
            fc = "#dce8f5"
        elif c > MID:
            fc = "#fdf0d5"
        else:
            fc = "#e8f5e5"
        ax.add_patch(mpatches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            facecolor=fc, edgecolor="#aaaaaa", linewidth=0.4
        ))

for (r, c, color, label) in OBJECTS:
    cy = N - 1 - r
    ax.add_patch(plt.Circle((c, cy), 0.32, color=color, zorder=3))
    if label and label != "TRUE":
        ax.text(c, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)
    if label == "TRUE":
        ring = plt.Circle((c, cy), 0.46, color="#f5c518", fill=False,
                          linewidth=2.5, linestyle="--", zorder=2)
        ax.add_patch(ring)

ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout(pad=0)
out = "/home/zhuyf/workspace/AskOrAct/results/figures/exploratory/gridworld_clean.png"
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"Saved: {out}")
