"""
Controlled instruction templates. Instruction u is a template string; candidate goals
are all obj_ids consistent with u. No NLP — deterministic template matching.
"""

from src.env import OBJ_TYPES, OBJ_COLORS, ROOM_NAMES

TEMPLATES = [
    ("get {}", ["type"]),
    ("get {} {}", ["color", "type"]),
    ("get {} in {}", ["type", "room"]),
    ("get {} object", ["color"]),
]


def _room_str(room_id):
    return ROOM_NAMES[room_id]


def _format_template(template_id, **kwargs):
    fmt, keys = TEMPLATES[template_id]
    parts = []
    for k in keys:
        if k == "room":
            parts.append(_room_str(kwargs["room_id"]))
        else:
            parts.append(kwargs[k])
    return fmt.format(*parts)


def sample_instruction(true_goal, world, template_id, rng):
    obj = world.get_object_by_id(true_goal)
    if obj is None:
        raise ValueError("true_goal obj_id not in world")
    if isinstance(obj, dict):
        type_, color, room_id = obj["type"], obj["color"], obj["room_id"]
    else:
        type_, color, room_id = obj.type, obj.color, obj.room_id
    return _format_template(template_id, type=type_, color=color, room_id=room_id)


def instruction_to_candidate_goals(u, world):
    u = u.strip().lower()
    candidates = []
    state = world.get_state()
    objects = state["objects"]
    for obj in objects:
        if obj.collected:
            continue
        obj_id = obj.obj_id
        type_, color, room_id = obj.type, obj.color, obj.room_id
        if u == _format_template(0, type=type_, color=color, room_id=room_id):
            candidates.append(obj_id)
        elif u == _format_template(1, type=type_, color=color, room_id=room_id):
            candidates.append(obj_id)
        elif u == _format_template(2, type=type_, color=color, room_id=room_id):
            candidates.append(obj_id)
        elif u == _format_template(3, type=type_, color=color, room_id=room_id):
            candidates.append(obj_id)
    return candidates


def get_template_id_for_ambiguity(u):
    u = u.strip().lower()
    if " in " in u and "object" not in u:
        return 2
    if u.endswith(" object"):
        return 3
    parts = u.split()
    if len(parts) == 2 and parts[0] == "get":
        return 0
    if len(parts) == 3 and parts[0] == "get":
        return 1
    return 0
