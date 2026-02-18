"""Assistant policies: AskOrAct, NeverAsk, AlwaysAsk. BFS for shortest path and distances."""

from collections import deque
import numpy as np
from src.env import ACTIONS, ACTION_DELTAS, grid_distance
from src.inference import posterior_entropy
from src.world import list_questions, answer_likelihood
from src import config


def _bfs_next_step(N, walls, from_pos, to_pos):
    if from_pos == to_pos:
        return "stay"
    q = deque([(from_pos, [])])
    seen = {from_pos}
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    action_names = ["right", "left", "down", "up"]
    while q:
        (r, c), path = q.popleft()
        for (dr, dc), a in zip(deltas, action_names):
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls or (nr, nc) in seen:
                continue
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if (nr, nc) == to_pos:
                return a
            seen.add((nr, nc))
            q.append(((nr, nc), path + [a]))
    return "stay"


def assistant_distance_to_goal(env, state, g):
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return float("inf")
    return grid_distance(env.N, env.walls, state["assistant_pos"], obj["pos"])


def principal_distance_to_goal(env, state, g):
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return float("inf")
    return grid_distance(env.N, env.walls, state["principal_pos"], obj["pos"])


def assistant_task_action(env, state, target_goal):
    """Pick whenever on any object cell; otherwise move one step toward target_goal."""
    a_pos = state["assistant_pos"]
    obj_by_pos = state.get("obj_by_pos", env.get_state().get("obj_by_pos", {}))
    if a_pos in obj_by_pos:
        return "pick"
    obj = env.get_object_by_id(target_goal)
    if obj is None or obj.get("collected", False):
        return "stay"
    return _bfs_next_step(env.N, env.walls, a_pos, obj["pos"])


def V_after_action_approx(env, state, g, assistant_action):
    a_pos = state["assistant_pos"]
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return float("inf")
    goal_pos = obj["pos"]
    if assistant_action == "pick":
        if a_pos == goal_pos:
            return 0
        return min(assistant_distance_to_goal(env, state, g), principal_distance_to_goal(env, state, g))
    dr, dc = ACTION_DELTAS.get(assistant_action, (0, 0))
    r, c = a_pos
    nr, nc = r + dr, c + dc
    if (nr, nc) in env.walls or not (0 <= nr < env.N and 0 <= nc < env.N):
        new_a_pos = a_pos
    else:
        new_a_pos = (nr, nc)
    dA = grid_distance(env.N, env.walls, new_a_pos, goal_pos)
    dP = principal_distance_to_goal(env, state, g)
    return min(dA, dP)


def CostAct(state, posterior, candidate_goals, env, answer_noise):
    if not candidate_goals or not posterior:
        return 1.0
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    task_act = assistant_task_action(env, state, g_star)
    V = sum(posterior.get(g, 0) * V_after_action_approx(env, state, g, task_act) for g in candidate_goals if posterior.get(g, 0) > 0)
    return 1.0 + V


def CostAsk(state, posterior, candidate_goals, env, answer_noise, question_cost=None):
    question_cost = question_cost or config.QUESTION_COST
    if not candidate_goals or not posterior:
        return 1.0 + question_cost
    questions = list_questions()
    best_cost = float("inf")
    for q in questions:
        expected_V = 0.0
        for ans in q[1]:
            P_ans = sum(posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals)
            if P_ans <= 0:
                continue
            p_new = {g: posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals}
            Z = sum(p_new.values())
            if Z <= 0:
                continue
            for g in candidate_goals:
                p_new[g] /= Z
            g_star = max(candidate_goals, key=lambda g: p_new.get(g, 0))
            task_act = assistant_task_action(env, state, g_star)
            V_ans = sum(p_new.get(g, 0) * V_after_action_approx(env, state, g, task_act) for g in candidate_goals)
            expected_V += P_ans * V_ans
        cost_q = 1.0 + question_cost + expected_V
        best_cost = min(best_cost, cost_q)
    return best_cost


def policy_never_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                    rng, beta, eps, answer_noise, **kwargs):
    if not candidate_goals or not posterior:
        return "stay", posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior


def policy_always_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                      questions_asked, rng, beta, eps, answer_noise, max_questions=None,
                      entropy_threshold=None, **kwargs):
    max_questions = max_questions or config.MAX_QUESTIONS
    entropy_threshold = entropy_threshold or config.ENTROPY_THRESHOLD
    if not candidate_goals or not posterior:
        return "stay", posterior
    ent = posterior_entropy(posterior)
    if questions_asked < max_questions and ent > entropy_threshold:
        questions = list_questions()
        best_q = None
        best_cost = float("inf")
        for q in questions:
            cost = 1.0 + config.QUESTION_COST
            for ans in q[1]:
                P_ans = sum(posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals)
                if P_ans <= 0:
                    continue
                p_new = {g: posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals}
                Z = sum(p_new.values())
                if Z > 0:
                    for g in candidate_goals:
                        p_new[g] /= Z
                    g_star = max(candidate_goals, key=lambda g: p_new.get(g, 0))
                    task_act = assistant_task_action(env, state, g_star)
                    cost += P_ans * sum(p_new.get(g, 0) * V_after_action_approx(env, state, g, task_act) for g in candidate_goals)
            if cost < best_cost:
                best_cost = cost
                best_q = q[0]
        if best_q is not None:
            return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior


def policy_ask_or_act(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                      questions_asked, rng, beta, eps, answer_noise, question_cost=None, **kwargs):
    question_cost = question_cost or config.QUESTION_COST
    if not candidate_goals or not posterior:
        return "stay", posterior
    cost_act = CostAct(state, posterior, candidate_goals, env, answer_noise)
    cost_ask = CostAsk(state, posterior, candidate_goals, env, answer_noise, question_cost)
    if cost_ask < cost_act:
        questions = list_questions()
        best_q = None
        best_cost = float("inf")
        for q in questions:
            cost = 1.0 + question_cost
            for ans in q[1]:
                P_ans = sum(posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals)
                if P_ans <= 0:
                    continue
                p_new = {g: posterior.get(g, 0) * answer_likelihood(q, ans, g, env, answer_noise) for g in candidate_goals}
                Z = sum(p_new.values())
                if Z > 0:
                    for g in candidate_goals:
                        p_new[g] /= Z
                    g_star = max(candidate_goals, key=lambda g: p_new.get(g, 0))
                    task_act = assistant_task_action(env, state, g_star)
                    cost += P_ans * sum(p_new.get(g, 0) * V_after_action_approx(env, state, g, task_act) for g in candidate_goals)
            if cost < best_cost:
                best_cost = cost
                best_q = q[0]
        if best_q is not None:
            return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior
