"""
GridWorldEnv: tiny gridworld with walls, optional two rooms, objects, principal and assistant.
"""

from collections import namedtuple, deque
import numpy as np
from src import config

# Object attributes
OBJ_TYPES = ("key", "gem", "coin")
OBJ_COLORS = ("red", "blue", "green")
ROOM_NAMES = ("room0", "room1")

# Actions: 0=up, 1=down, 2=left, 3=right, 4=stay, 5=pick
ACTIONS = ("up", "down", "left", "right", "stay", "pick")
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
    "pick": (0, 0),
}

ObjectInfo = namedtuple("ObjectInfo", ["obj_id", "type", "color", "room_id", "pos", "collected"])  # immutable snapshot


def _mutable_obj(obj_id, type_, color, room_id, pos):
    """Mutable object dict for env state."""
    return {"obj_id": obj_id, "type": type_, "color": color, "room_id": room_id, "pos": pos, "collected": False}


def grid_distance(N, walls, from_pos, to_pos):
    """Shortest path length (BFS) from from_pos to to_pos; returns float('inf') if unreachable."""
    if from_pos == to_pos:
        return 0
    q = deque([(from_pos, 0)])
    seen = {from_pos}
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while q:
        (r, c), d = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls or (nr, nc) in seen:
                continue
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if (nr, nc) == to_pos:
                return d + 1
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return float("inf")


class GridWorldEnv:
    """
    Grid of size N x N with boundary walls.
    Optionally split into 2 rooms by a wall with a single doorway.
    Objects have obj_id, type, color, room_id. Principal and assistant are agents.
    """

    def __init__(self, N, objects, principal_pos, assistant_pos, walls=None, doorway=None):
        """
        objects: list of dicts with keys obj_id, type, color, room_id, pos (r,c)
        walls: set of (r,c) that are walls (default: boundary only)
        doorway: (r,c) single cell connecting two rooms, or None
        """
        self.N = N
        self.doorway = doorway
        # Build wall set: boundary + optional internal wall (all except doorway)
        self.walls = set()
        for r in range(N):
            self.walls.add((r, 0))
            self.walls.add((r, N - 1))
        for c in range(N):
            self.walls.add((0, c))
            self.walls.add((N - 1, c))
        if walls:
            self.walls.update(walls)

        # Object list: mutable dicts so we can set collected=True
        self.objects = []
        for o in objects:
            self.objects.append(_mutable_obj(
                o["obj_id"], o["type"], o["color"], o["room_id"], tuple(o["pos"])
            ))

        self.principal_pos = tuple(principal_pos)
        self.assistant_pos = tuple(assistant_pos)
        self._obj_by_pos = {}
        self._rebuild_obj_by_pos()
        self._initial_state = None  # set by worldgen for reset(seed)

    def _rebuild_obj_by_pos(self):
        self._obj_by_pos = {}
        for o in self.objects:
            if not o["collected"]:
                self._obj_by_pos[o["pos"]] = o

    def get_state(self):
        """Return current state for inference/rendering: positions, object map."""
        return {
            "principal_pos": self.principal_pos,
            "assistant_pos": self.assistant_pos,
            "objects": [ObjectInfo(o["obj_id"], o["type"], o["color"], o["room_id"], o["pos"], o["collected"]) for o in self.objects],
            "obj_by_pos": dict(self._obj_by_pos),
        }

    def get_object_by_id(self, obj_id):
        for o in self.objects:
            if o["obj_id"] == obj_id:
                return o
        return None

    def _move(self, pos, action):
        if action == "pick":
            return pos
        dr, dc = ACTION_DELTAS[action]
        r, c = pos
        nr, nc = r + dr, c + dc
        if (nr, nc) in self.walls:
            return pos
        return (nr, nc)

    def _free_cell(self, pos):
        return 0 <= pos[0] < self.N and 0 <= pos[1] < self.N and pos not in self.walls

    def step(self, principal_action, assistant_action, true_goal_obj_id=None, wrong_pick_fail=None):
        """
        Apply principal then assistant action. Return (state, done, info).
        When true_goal_obj_id is provided, info includes success, assistant_picked_goal,
        picked_obj_id, principal_on_goal_cell. done=True when assistant picks the true goal.
        """
        # Principal: pick has no effect when PRINCIPAL_CAN_PICK is False
        eff_p_action = principal_action
        if principal_action == "pick" and not getattr(config, "PRINCIPAL_CAN_PICK", True):
            eff_p_action = "stay"
        new_p_pos = self._move(self.principal_pos, eff_p_action)
        if self._free_cell(new_p_pos):
            self.principal_pos = new_p_pos
        if eff_p_action == "pick":
            o = self._obj_by_pos.get(self.principal_pos)
            if o is not None:
                o["collected"] = True
                self._rebuild_obj_by_pos()

        goal_pos = None
        if true_goal_obj_id is not None:
            go = self.get_object_by_id(true_goal_obj_id)
            if go and not go.get("collected", False):
                goal_pos = go["pos"]
        principal_on_goal_cell = (goal_pos is not None and self.principal_pos == goal_pos)

        # Assistant moves second
        new_a_pos = self._move(self.assistant_pos, assistant_action)
        if self._free_cell(new_a_pos):
            self.assistant_pos = new_a_pos

        picked_obj_id = None
        assistant_picked_goal = False
        wrong_pick = False
        terminated_by_wrong_pick = False
        done = False
        if wrong_pick_fail is None:
            wrong_pick_fail = getattr(config, "WRONG_PICK_FAIL", False)
        if assistant_action == "pick":
            o = self._obj_by_pos.get(self.assistant_pos)
            if o is not None:
                picked_obj_id = o["obj_id"]
                o["collected"] = True
                self._rebuild_obj_by_pos()
                if true_goal_obj_id is not None and picked_obj_id == true_goal_obj_id:
                    assistant_picked_goal = True
                    done = True
                elif true_goal_obj_id is not None:
                    wrong_pick = True
                    if wrong_pick_fail:
                        done = True
                        terminated_by_wrong_pick = True

        success = assistant_picked_goal
        info = {
            "step_cost": 1,
            "success": success,
            "picked_obj_id": picked_obj_id,
            "assistant_picked_goal": assistant_picked_goal,
            "wrong_pick": wrong_pick,
            "terminated_by_wrong_pick": terminated_by_wrong_pick,
            "principal_on_goal_cell": principal_on_goal_cell,
        }
        return self.get_state(), done, info

    def check_goal_satisfied(self, true_goal_obj_id):
        """True if the object with true_goal_obj_id has been collected."""
        obj = self.get_object_by_id(true_goal_obj_id)
        return obj is not None and obj["collected"]

    def set_initial_state(self, principal_pos, assistant_pos):
        """Store initial positions for reset(). Call after construction (e.g. by worldgen)."""
        self._initial_state = (tuple(principal_pos), tuple(assistant_pos))

    def reset(self, seed=None):
        """Reset env to initial state: initial positions, no objects collected."""
        if seed is not None:
            np.random.seed(seed)
        if self._initial_state is not None:
            self.principal_pos = self._initial_state[0]
            self.assistant_pos = self._initial_state[1]
        for o in self.objects:
            o["collected"] = False
        self._rebuild_obj_by_pos()
        return self.get_state()

    def reset_from_state(self, principal_pos, assistant_pos, objects_collected):
        """Reset positions and object collected flags. objects_collected: list of bool per object."""
        self.principal_pos = tuple(principal_pos)
        self.assistant_pos = tuple(assistant_pos)
        for i, o in enumerate(self.objects):
            self.objects[i]["collected"] = objects_collected[i]
        self._rebuild_obj_by_pos()

    def render_ascii(self, true_goal_obj_id=None):
        """Print grid with P=principal, A=assistant, objects as type letter, walls #."""
        grid = [[" " for _ in range(self.N)] for _ in range(self.N)]
        for (r, c) in self.walls:
            if 0 <= r < self.N and 0 <= c < self.N:
                grid[r][c] = "#"
        for o in self.objects:
            if o["collected"]:
                continue
            r, c = o["pos"]
            letter = o["type"][0].upper()  # K, G, C
            if true_goal_obj_id is not None and o["obj_id"] == true_goal_obj_id:
                letter = letter.lower()  # target as lowercase
            if grid[r][c] == " ":
                grid[r][c] = letter
            else:
                grid[r][c] = "?"
        r, c = self.principal_pos
        grid[r][c] = "P"
        r, c = self.assistant_pos
        grid[r][c] = "A"
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)
