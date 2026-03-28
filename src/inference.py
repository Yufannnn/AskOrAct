"""
Bayesian inverse planning: posterior over goals from observed principal actions.
"""

import numpy as np


def init_posterior(candidate_goals, prior_type="uniform", env=None, assistant_pos=None):
    """Prior over candidate goals. Return dict obj_id -> prob."""
    if not candidate_goals:
        return {}
    n = len(candidate_goals)
    if prior_type == "distance" and env is not None and assistant_pos is not None:
        from src.env import grid_distance
        weights = {}
        for g in candidate_goals:
            obj = env.get_object_by_id(g)
            if obj is None:
                weights[g] = 1.0
            else:
                d = grid_distance(env.N, env.walls, assistant_pos, obj["pos"])
                if d == float("inf"):
                    d = 2 * env.N
                weights[g] = 1.0 / (1.0 + d)
        z = sum(weights.values())
        if z > 0:
            return {g: w / z for g, w in weights.items()}
    return {g: 1.0 / n for g in candidate_goals}


def update_posterior(posterior, state, observed_action, candidate_goals, env, beta, eps):
    """
    p_{t+1}(g) ∝ p_t(g) * P(a_t | s_t, g). Normalize. Use log-space for numerical stability.
    posterior: dict obj_id -> probability (updated in place and returned).
    """
    if not candidate_goals or not posterior:
        return posterior
    from src.agents.principal import principal_action_probs
    log_post = {}
    for g in candidate_goals:
        prev = posterior.get(g, 0)
        if prev <= 0:
            continue
        probs = principal_action_probs(state, g, env, beta, eps)
        p_a = probs.get(observed_action, 1e-10)
        log_post[g] = np.log(prev + 1e-15) + np.log(p_a)
    if not log_post:
        return posterior
    max_lp = max(log_post.values())
    Z = sum(np.exp(lp - max_lp) for lp in log_post.values())
    for g in candidate_goals:
        posterior[g] = np.exp(log_post[g] - max_lp) / Z if g in log_post else 0.0
    return posterior


def posterior_entropy(posterior):
    """Entropy of posterior distribution. Higher = more uncertainty."""
    probs = [p for p in posterior.values() if p > 0]
    if not probs:
        return 0.0
    probs = np.array(probs)
    return -np.sum(probs * np.log(probs + 1e-15))
