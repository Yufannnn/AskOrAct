"""Build setup_overview.excalidraw with the gridworld PNG embedded plus
editable annotations on top (title, side info boxes, "true goal" callout)."""
import base64
import json
import time
from pathlib import Path

ROOT = Path("/home/zhuyf/workspace/AskOrAct")
PNG = ROOT / "results/figures/exploratory/gridworld_clean.png"
OUT = ROOT / "presentation/setup_overview.excalidraw"

png_b64 = base64.b64encode(PNG.read_bytes()).decode("ascii")
data_url = f"data:image/png;base64,{png_b64}"

now_ms = int(time.time() * 1000)
file_id = "grid_clean_v1"

# ── canvas plan (px) ──────────────────────────────────────────────────────
# Image: (110, 130) size 500x500    (extra left margin for "door" label)
# Side boxes: (640, 130) width 460, height 130, vertical gap 20
# Title + subtitle at top
# "true goal" callout: in upper-right of right room, arrow down to dashed ring
IMG_X, IMG_Y, IMG_W, IMG_H = 110, 130, 500, 500

# True-goal pixel (relative to 9x9 grid, col 7, row 5, centre)
# In display coords: x = IMG_X + (col + 0.5)/9 * IMG_W  (matplotlib uses cell centres)
TG_X = IMG_X + (7 + 0.5) / 9 * IMG_W
TG_Y = IMG_Y + (5 + 0.5) / 9 * IMG_H

elements = []


def add(el):
    el.setdefault("angle", 0)
    el.setdefault("strokeColor", "#1e1e1e")
    el.setdefault("backgroundColor", "transparent")
    el.setdefault("fillStyle", "solid")
    el.setdefault("strokeWidth", 2)
    el.setdefault("strokeStyle", "solid")
    el.setdefault("roughness", 1)
    el.setdefault("opacity", 100)
    el.setdefault("groupIds", [])
    el.setdefault("frameId", None)
    el.setdefault("roundness", None)
    el.setdefault("seed", len(elements) + 1000)
    el.setdefault("version", 1)
    el.setdefault("versionNonce", len(elements) + 1000)
    el.setdefault("isDeleted", False)
    el.setdefault("boundElements", None)
    el.setdefault("updated", now_ms)
    el.setdefault("link", None)
    el.setdefault("locked", False)
    elements.append(el)
    return el


def text(id_, x, y, w, h, txt, size=15, color="#1e1e1e",
         align="left", valign="top", container=None, family=1,
         auto_resize=False):
    add({
        "type": "text", "id": id_,
        "x": x, "y": y, "width": w, "height": h,
        "strokeColor": color, "strokeWidth": 1,
        "text": txt, "fontSize": size, "fontFamily": family,
        "textAlign": align, "verticalAlign": valign,
        "containerId": container, "originalText": txt,
        "boundElements": [],
        "autoResize": auto_resize,
        "lineHeight": 1.25,
    })


def rect(id_, x, y, w, h, fc, ec, label=None, label_size=14):
    add({
        "type": "rectangle", "id": id_,
        "x": x, "y": y, "width": w, "height": h,
        "strokeColor": ec, "backgroundColor": fc,
        "roundness": {"type": 3},
        "boundElements": ([{"type": "text", "id": id_ + "_t"}] if label else []),
    })
    if label:
        text(id_ + "_t", x, y, w, h, label,
             size=label_size, align="center", valign="middle",
             container=id_)


def arrow(id_, x, y, points, color="#1e1e1e", dashed=False, label=None,
          start_binding=None, end_binding=None):
    el = {
        "type": "arrow", "id": id_,
        "x": x, "y": y,
        "width": max(abs(p[0]) for p in points) or 1,
        "height": max(abs(p[1]) for p in points) or 1,
        "strokeColor": color, "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid",
        "roundness": {"type": 2},
        "points": points,
        "lastCommittedPoint": None,
        "startArrowhead": None, "endArrowhead": "arrow",
        "boundElements": ([{"type": "text", "id": id_ + "_lbl"}] if label else []),
    }
    if start_binding:
        el["startBinding"] = start_binding
    if end_binding:
        el["endBinding"] = end_binding
    add(el)
    if label:
        text(id_ + "_lbl", x, y + 8, 180, 18, label,
             size=12, color=color, align="center", valign="middle",
             container=id_)


# ── 1. Image element ──────────────────────────────────────────────────────
add({
    "type": "image", "id": "img_grid",
    "x": IMG_X, "y": IMG_Y, "width": IMG_W, "height": IMG_H,
    "strokeColor": "transparent", "backgroundColor": "transparent",
    "strokeWidth": 1, "roughness": 0,
    "fileId": file_id, "scale": [1, 1], "status": "saved", "crop": None,
})

# ── 2. Title + subtitle (Virgil) ──────────────────────────────────────────
text("t_title", 110, 40, 980, 36,
     "Task setup: ambiguous instruction, hidden goal, shared world state",
     size=22, family=1)
text("t_sub", 110, 78, 980, 22,
     "Example shown for K = 3 — three red objects share the surface "
     "instruction, but only one is the principal's true goal.",
     size=13, color="#555555", family=1)

# ── 3. Side info boxes ────────────────────────────────────────────────────
BX, BW, BH, GAP = 640, 320, 100, 18
rect("b_inst", BX, 150, BW, BH, "#fff3bf", "#f59e0b")
text("t_inst_h", BX + 14, 162, BW - 28, 20, "Instruction",
     size=13, color="#8a5a00", family=1)
text("t_inst_b", BX + 14, 186, BW - 28, 60,
     '"get red object"\nK = 3 candidates share this surface form',
     size=11, family=1)

rect("b_obs", BX, 150 + BH + GAP, BW, BH, "#a5d8ff", "#4a9eed")
text("t_obs_h", BX + 14, 162 + BH + GAP, BW - 28, 20, "Assistant observes",
     size=13, color="#1e4d8a", family=1)
text("t_obs_b", BX + 14, 186 + BH + GAP, BW - 28, 60,
     "• world state + instruction\n• principal's actions over time",
     size=11, family=1)

rect("b_ask", BX, 150 + 2 * (BH + GAP), BW, BH, "#b2f2bb", "#22c55e")
text("t_ask_h", BX + 14, 162 + 2 * (BH + GAP), BW - 28, 20, "Clarification menu",
     size=13, color="#1f6b2b", family=1)
text("t_ask_b", BX + 14, 186 + 2 * (BH + GAP), BW - 28, 60,
     "ask_color   ask_type   ask_room\nanswers update the posterior over goals",
     size=11, family=1)

# ── 4. "true goal" callout (text + arrow) ─────────────────────────────────
# Place text floating ABOVE the gridworld image, just to the right of centre,
# so the arrow drops cleanly into the dashed ring on the right side.
TG_TXT_X, TG_TXT_Y, TG_TXT_W = 330, 95, 240
text("t_truegoal", TG_TXT_X, TG_TXT_Y, TG_TXT_W, 36,
     "true goal\n(hidden from assistant)",
     size=14, color="#c0392b", align="center", family=1)
# Arrow from bottom-centre of the callout text down to the dashed ring.
arrow_x = TG_TXT_X + TG_TXT_W / 2
arrow_y = TG_TXT_Y + 38
arrow("a_truegoal", arrow_x, arrow_y,
      [[0, 0], [TG_X - arrow_x, TG_Y - arrow_y]],
      color="#c0392b")

# ── 5. "door" label + arrow to the doorway cell ───────────────────────────
DOOR_X = IMG_X + (4 + 0.5) / 9 * IMG_W
DOOR_Y = IMG_Y + (4 + 0.5) / 9 * IMG_H
text("t_door", 25, 370, 75, 22, "door",
     size=14, color="#3d7a3d", family=1)
# Arrow from right edge of "door" label rightward into the doorway cell.
arrow("a_door", 100, 380,
      [[0, 0], [DOOR_X - 110, DOOR_Y - 380]],
      color="#3d7a3d")

# ── assemble file ─────────────────────────────────────────────────────────
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {
        file_id: {
            "mimeType": "image/png",
            "id": file_id,
            "dataURL": data_url,
            "created": now_ms,
            "lastRetrieved": now_ms,
        }
    },
}

OUT.write_text(json.dumps(doc, indent=2))
print(f"Wrote {OUT}  ({len(elements)} elements, image {len(png_b64)//1024} KB b64)")
