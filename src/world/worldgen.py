"""
World generation: grid with walls, two rooms, M objects. Ensures exactly K objects
match the emitted instruction u.
"""

import numpy as np
from src.env import GridWorldEnv, OBJ_TYPES, OBJ_COLORS
from src.world.instructions import instruction_to_candidate_goals, TEMPLATES
from src import config


def _room_str(room_id):
    return "room0" if room_id == 0 else "room1"


def _free_cells(N, walls):
    out = []
    for r in range(1, N - 1):
        for c in range(1, N - 1):
            if (r, c) not in walls:
                out.append((r, c))
    return out


def _build_walls_and_doorway(N):
    walls = set()
    for r in range(N):
        walls.add((r, 0))
        walls.add((r, N - 1))
    for c in range(N):
        walls.add((0, c))
        walls.add((N - 1, c))
    doorway = None
    if config.TWO_ROOMS and N >= 5:
        mid = N // 2
        for r in range(1, N - 1):
            if r != mid:
                walls.add((r, mid))
        doorway = (mid, mid)
    return walls, doorway


def generate_world(seed, N=None, M=None, ambiguity_K=2):
    """Returns (env, instruction_u, true_goal_obj_id)."""
    rng = np.random.default_rng(seed)
    N = N or config.DEFAULT_N
    M = M or config.DEFAULT_M
    K = min(ambiguity_K, M)
    walls, doorway = _build_walls_and_doorway(N)
    free = _free_cells(N, walls)
    if len(free) < M + 2:
        raise ValueError("Not enough free cells for M objects and 2 agents")

    template_id = int(rng.integers(0, 4))
    fmt, keys = TEMPLATES[template_id]
    true_type = rng.choice(OBJ_TYPES)
    true_color = rng.choice(OBJ_COLORS)
    true_room_id = rng.integers(0, 2) if doorway else 0

    def matches(o_type, o_color, o_room):
        if template_id == 0:
            return o_type == true_type
        if template_id == 1:
            return o_type == true_type and o_color == true_color
        if template_id == 2:
            return o_type == true_type and o_room == true_room_id
        if template_id == 3:
            return o_color == true_color
        return False

    objects = []
    for i in range(K):
        objects.append({
            "obj_id": i,
            "type": true_type,
            "color": true_color,
            "room_id": true_room_id,
            "pos": None,
        })
    for i in range(K, M):
        o_type = rng.choice(OBJ_TYPES)
        o_color = rng.choice(OBJ_COLORS)
        o_room = rng.integers(0, 2) if doorway else 0
        while matches(o_type, o_color, o_room):
            o_type = rng.choice(OBJ_TYPES)
            o_color = rng.choice(OBJ_COLORS)
            o_room = rng.integers(0, 2) if doorway else 0
        objects.append({"obj_id": i, "type": o_type, "color": o_color, "room_id": o_room, "pos": None})

    rng.shuffle(objects)
    for i, o in enumerate(objects):
        o["obj_id"] = i
    true_goal_obj_id = next(i for i, o in enumerate(objects) if o["type"] == true_type and o["color"] == true_color and o["room_id"] == true_room_id)

    used = set()
    for o in objects:
        idx = rng.integers(0, len(free))
        while free[idx] in used:
            idx = rng.integers(0, len(free))
        o["pos"] = free[idx]
        used.add(free[idx])
    principal_pos = next((free[i] for i in range(len(free)) if free[i] not in used), None)
    if principal_pos is not None:
        used.add(principal_pos)
    assistant_pos = next((free[i] for i in range(len(free)) if free[i] not in used), None)
    if assistant_pos is not None:
        used.add(assistant_pos)
    if principal_pos is None or assistant_pos is None:
        raise ValueError("Could not place both agents")

    env = GridWorldEnv(N, objects, principal_pos, assistant_pos, walls=walls, doorway=doorway)
    env.set_initial_state(principal_pos, assistant_pos)

    true_obj = env.get_object_by_id(true_goal_obj_id)
    if template_id == 0:
        u = "get {}".format(true_obj["type"])
    elif template_id == 1:
        u = "get {} {}".format(true_obj["color"], true_obj["type"])
    elif template_id == 2:
        u = "get {} in {}".format(true_obj["type"], "room0" if true_obj["room_id"] == 0 else "room1")
    else:
        u = "get {} object".format(true_obj["color"])

    candidates = instruction_to_candidate_goals(u, env)
    if len(candidates) != K:
        return generate_world(seed + 1, N, M, ambiguity_K)
    return env, u, true_goal_obj_id
