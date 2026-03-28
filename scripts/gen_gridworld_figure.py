"""Generate a clean two-room gridworld figure for the final report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

N = 9
MID = N // 2  # col 4
DOORWAY = (MID, MID)  # row 4, col 4

# Example object placements: (row, col, type, color)
OBJECTS = [
    (2, 2, 'gem',  'red'),    # candidate goal 1 (red gem, room0)
    (6, 2, 'key',  'red'),    # candidate goal 2 (red key, room0)
    (2, 6, 'coin', 'blue'),   # distractor (room1)
    (5, 7, 'gem',  'green'),  # distractor (room1)
    (7, 6, 'key',  'blue'),   # distractor (room1)
    (4, 1, 'coin', 'green'),  # distractor (room0)
]
# Principal position, assistant position
PRINCIPAL_POS = (3, 3)
ASSISTANT_POS = (4, 6)

OBJ_SYMBOLS = {'key': '🔑', 'gem': '💎', 'coin': '⬡'}
OBJ_LATEX = {'key': 'K', 'gem': 'G', 'coin': 'C'}

COLOR_MAP = {
    'red':   '#e74c3c',
    'blue':  '#3498db',
    'green': '#2ecc71',
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [2, 1]})

# ---- Left panel: gridworld ----
ax = axes[0]
ax.set_aspect('equal')
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(-0.5, N - 0.5)

# Draw cells
for r in range(N):
    for c in range(N):
        is_outer_wall = (r == 0 or r == N-1 or c == 0 or c == N-1)
        is_mid_wall = (c == MID and r != MID and r not in (0, N-1))
        if is_outer_wall or is_mid_wall:
            fc = '#555555'
        elif c < MID:
            fc = '#dce8f5'
        elif c > MID:
            fc = '#fdf0d5'
        else:  # doorway column (only doorway row is here)
            fc = '#e8f5e5'
        rect = mpatches.FancyBboxPatch(
            (c - 0.5, r - 0.5), 1, 1,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor='#aaaaaa', linewidth=0.4
        )
        ax.add_patch(rect)

# Draw room labels
ax.text(2, 0.2, 'Room 0', ha='center', va='center', fontsize=9,
        fontweight='bold', color='#2c5f8a')
ax.text(6.5, 0.2, 'Room 1', ha='center', va='center', fontsize=9,
        fontweight='bold', color='#8a6a2c')

# Draw doorway label
ax.annotate('doorway', xy=(MID, MID), xytext=(MID + 1.8, MID - 1.5),
            fontsize=7.5, color='#3d7a3d',
            arrowprops=dict(arrowstyle='->', color='#3d7a3d', lw=1.2),
            ha='center')

# Draw objects
for (r, c, otype, ocolor) in OBJECTS:
    circle = plt.Circle((c, N - 1 - r), 0.32, color=COLOR_MAP[ocolor], zorder=3)
    ax.add_patch(circle)
    ax.text(c, N - 1 - r, OBJ_LATEX[otype], ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)

# Highlight K=2 candidates (both red objects)
for (r, c, otype, ocolor) in OBJECTS[:2]:
    ring = plt.Circle((c, N - 1 - r), 0.42, color='gold', fill=False,
                       linewidth=2.5, zorder=2)
    ax.add_patch(ring)

# Draw principal
pr, pc = PRINCIPAL_POS
ax.add_patch(plt.Circle((pc, N - 1 - pr), 0.35, color='#9b59b6', zorder=3))
ax.text(pc, N - 1 - pr, 'P', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', zorder=4)

# Draw assistant
ar, ac = ASSISTANT_POS
ax.add_patch(plt.Circle((ac, N - 1 - ar), 0.35, color='#e67e22', zorder=3))
ax.text(ac, N - 1 - ar, 'A', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', zorder=4)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('Two-Room $9\\times9$ Gridworld', fontsize=11, pad=8)

# ---- Right panel: legend / info box ----
ax2 = axes[1]
ax2.axis('off')

lines = [
    (0.05, 0.95, 'Instruction (ambiguous):', 'black', 9.5, 'bold'),
    (0.05, 0.87, '  "Bring me the red object"', '#c0392b', 9, 'italic'),
    (0.05, 0.79, 'Candidate goals ($K{=}2$):', 'black', 9.5, 'bold'),
    (0.05, 0.72, '  red gem  (room 0)', COLOR_MAP['red'], 9, 'normal'),
    (0.05, 0.65, '  red key  (room 0)', COLOR_MAP['red'], 9, 'normal'),
    (0.05, 0.55, 'Questions available:', 'black', 9.5, 'bold'),
    (0.05, 0.48, '  Q1: What color?', '#2c2c2c', 8.5, 'normal'),
    (0.05, 0.42, '  Q2: What type?', '#2c2c2c', 8.5, 'normal'),
    (0.05, 0.36, '  Q3: Which room?', '#2c2c2c', 8.5, 'normal'),
]

for x, y, text, color, fs, style in lines:
    ax2.text(x, y, text, transform=ax2.transAxes, fontsize=fs,
             color=color, fontstyle=style if style == 'italic' else 'normal',
             fontweight=style if style == 'bold' else 'normal', va='top')

# Legend icons
legend_items = [
    (COLOR_MAP['red'],   'Red object (candidate)'),
    (COLOR_MAP['blue'],  'Blue object (distractor)'),
    (COLOR_MAP['green'], 'Green object (distractor)'),
    ('#9b59b6', 'Principal (P)'),
    ('#e67e22', 'Assistant (A)'),
]
for i, (c, label) in enumerate(legend_items):
    yy = 0.23 - i * 0.065
    circle = plt.Circle((0.06, yy), 0.025, color=c,
                         transform=ax2.transAxes, clip_on=False)
    ax2.add_patch(circle)
    ax2.text(0.13, yy, label, transform=ax2.transAxes,
             fontsize=8, va='center', color='#222222')

ax2.set_title('', fontsize=1)

plt.tight_layout(pad=1.2)
out_path = '/home/zhuyf/workspace/AskOrAct/results/gridworld_layout.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
print(f"Saved: {out_path}")
