"""Principal policy: approximately rational toward true goal. Stochastic with beta and eps."""

import numpy as np
from src.env import ACTIONS, ACTION_DELTAS, grid_distance
from src import config


def get_principal_actions():
    """Action set for principal: excludes pick when PRINCIPAL_CAN_PICK is False."""
    if getattr(config, "PRINCIPAL_CAN_PICK", True):
        return ACTIONS
    return tuple(a for a in ACTIONS if a != "pick")


def _goal_pos(env, g):
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return None
    return obj["pos"]


def principal_action_probs(state, g, env, beta, eps):
    """P(a | s, g) ∝ exp(-beta * d(s', g)) with eps uniform random. Uses only principal-allowed actions."""
    actions = get_principal_actions()
    goal_pos = _goal_pos(env, g)
    if goal_pos is None:
        return {a: 1.0 / len(actions) for a in actions}
    p_pos = state["principal_pos"]
    N, walls = env.N, env.walls
    dists = {}
    for a in actions:
        if a == "stay" or a == "pick":
            next_pos = p_pos
        else:
            dr, dc = ACTION_DELTAS[a]
            r, c = p_pos
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls or not (0 <= nr < N and 0 <= nc < N):
                next_pos = p_pos
            else:
                next_pos = (nr, nc)
        d = grid_distance(N, walls, next_pos, goal_pos)
        dists[a] = d
    max_lp = max(-beta * dists[a] for a in actions)
    probs = {}
    for a in actions:
        lp = -beta * dists[a]
        probs[a] = np.exp(lp - max_lp)
    Z = sum(probs.values())
    for a in actions:
        probs[a] = (1 - eps) * (probs[a] / Z) + eps / len(actions)
    return probs


def sample_principal_action(state, true_goal, env, rng, beta, eps):
    probs = principal_action_probs(state, true_goal, env, beta, eps)
    actions = list(probs.keys())
    p = [probs[a] for a in actions]
    return rng.choice(actions, p=p)
