"""
World generation: grid with walls, two rooms, M objects. Ensures exactly K objects
match the emitted instruction u.
"""

import numpy as np
from src.env import GridWorldEnv, OBJ_TYPES, OBJ_COLORS, grid_distance
from src.world.instructions import instruction_to_candidate_goals
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


def _room_id_from_pos(pos, N, doorway):
    if doorway is None:
        return 0
    mid = N // 2
    return 0 if pos[1] < mid else 1


def _candidate_positions_ok(candidate_positions, N, walls, doorway):
    if len(candidate_positions) <= 1:
        return True
    min_pairwise_dist = N // 2
    for i in range(len(candidate_positions)):
        for j in range(i + 1, len(candidate_positions)):
            d = grid_distance(N, walls, candidate_positions[i], candidate_positions[j])
            if d < min_pairwise_dist:
                return False
    if config.TWO_ROOMS and doorway is not None:
        rooms = {_room_id_from_pos(p, N, doorway) for p in candidate_positions}
        if len(rooms) < 2:
            return False
    return True


def _sample_candidate_positions(rng, free_cells, K, N, walls, doorway):
    if K <= 0:
        return []
    if K == 1:
        idx = int(rng.integers(0, len(free_cells)))
        return [free_cells[idx]]
    for _ in range(1000):
        ids = rng.choice(len(free_cells), size=K, replace=False)
        positions = [free_cells[i] for i in ids]
        if _candidate_positions_ok(positions, N, walls, doorway):
            return positions
    return None


def generate_world(seed, N=None, M=None, ambiguity_K=2):
    """Returns (env, instruction_u, true_goal_obj_id)."""
    N = N or config.DEFAULT_N
    M = M or config.DEFAULT_M
    K = min(ambiguity_K, M)
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        walls, doorway = _build_walls_and_doorway(N)
        free = _free_cells(N, walls)
        if len(free) < M + 2:
            raise ValueError("Not enough free cells for M objects and 2 agents")

        template_choices = [0, 1, 2, 3]
        if K >= 2 and config.TWO_ROOMS and doorway is not None and 2 in template_choices:
            template_choices.remove(2)  # room template cannot satisfy multi-room candidate constraint for K>=2
        template_id = int(rng.choice(template_choices))
        true_type = rng.choice(OBJ_TYPES)
        true_color = rng.choice(OBJ_COLORS)
        true_room_id = int(rng.integers(0, 2)) if doorway else 0

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
        for _ in range(K):
            if template_id == 0:
                o_type = true_type
                o_color = rng.choice(OBJ_COLORS)
                o_room = int(rng.integers(0, 2)) if doorway else 0
            elif template_id == 1:
                o_type = true_type
                o_color = true_color
                o_room = int(rng.integers(0, 2)) if doorway else 0
            elif template_id == 2:
                o_type = true_type
                o_color = rng.choice(OBJ_COLORS)
                o_room = true_room_id
            else:
                o_type = rng.choice(OBJ_TYPES)
                o_color = true_color
                o_room = int(rng.integers(0, 2)) if doorway else 0
            objects.append({
                "obj_id": -1,
                "type": o_type,
                "color": o_color,
                "room_id": o_room,
                "pos": None,
            })
        for _ in range(K, M):
            o_type = rng.choice(OBJ_TYPES)
            o_color = rng.choice(OBJ_COLORS)
            o_room = int(rng.integers(0, 2)) if doorway else 0
            while matches(o_type, o_color, o_room):
                o_type = rng.choice(OBJ_TYPES)
                o_color = rng.choice(OBJ_COLORS)
                o_room = int(rng.integers(0, 2)) if doorway else 0
            objects.append({"obj_id": -1, "type": o_type, "color": o_color, "room_id": o_room, "pos": None})

        candidate_idxs = [i for i, o in enumerate(objects) if matches(o["type"], o["color"], o["room_id"])]
        if len(candidate_idxs) != K:
            continue

        candidate_positions = _sample_candidate_positions(rng, free, K, N, walls, doorway)
        if candidate_positions is None:
            continue
        if template_id == 2 and doorway is not None:
            room_filtered = [p for p in candidate_positions if _room_id_from_pos(p, N, doorway) == true_room_id]
            if not room_filtered:
                continue
            candidate_positions = [room_filtered[0]]
        rng.shuffle(candidate_positions)

        used = set()
        for idx, pos in zip(candidate_idxs, candidate_positions):
            objects[idx]["pos"] = pos
            used.add(pos)

        rem_cells = [p for p in free if p not in used]
        rng.shuffle(rem_cells)
        rem_idx = 0
        for i, o in enumerate(objects):
            if o["pos"] is not None:
                continue
            if rem_idx >= len(rem_cells):
                break
            o["pos"] = rem_cells[rem_idx]
            used.add(rem_cells[rem_idx])
            rem_idx += 1
        if any(o["pos"] is None for o in objects):
            continue

        for o in objects:
            o["room_id"] = _room_id_from_pos(o["pos"], N, doorway)

        rng.shuffle(objects)
        for i, o in enumerate(objects):
            o["obj_id"] = i

        matching_ids = [o["obj_id"] for o in objects if matches(o["type"], o["color"], o["room_id"])]
        if len(matching_ids) != K:
            continue
        if K >= 2:
            candidate_pos = [o["pos"] for o in objects if matches(o["type"], o["color"], o["room_id"])]
            if not _candidate_positions_ok(candidate_pos, N, walls, doorway):
                continue
        true_goal_obj_id = int(rng.choice(matching_ids))

        rem_cells = [p for p in free if p not in used]
        if len(rem_cells) < 2:
            continue
        rng.shuffle(rem_cells)
        principal_pos, assistant_pos = rem_cells[0], rem_cells[1]

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
            continue
        if K >= 2:
            c_objs = [env.get_object_by_id(g) for g in candidates]
            c_positions = [o["pos"] for o in c_objs if o is not None]
            if len(c_positions) != K or not _candidate_positions_ok(c_positions, N, walls, doorway):
                continue
        return env, u, true_goal_obj_id

    raise RuntimeError(f"Failed to generate world with constraints for seed={seed}, K={K}")
