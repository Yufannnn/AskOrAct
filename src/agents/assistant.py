"""Assistant policies: AskOrAct, NeverAsk, AlwaysAsk, and baseline askers."""

from collections import deque
import math
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
                full_path = path + [a]
                return full_path[0] if full_path else "stay"
            seen.add((nr, nc))
            q.append(((nr, nc), path + [a]))
    return "stay"


def assistant_distance_to_goal(env, state, g):
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return float("inf")
    return grid_distance(env.N, env.walls, state["assistant_pos"], obj["pos"])


def _assistant_value_from_pos(env, pos, g):
    obj = env.get_object_by_id(g)
    if obj is None or obj.get("collected", False):
        return float("inf")
    d = grid_distance(env.N, env.walls, pos, obj["pos"])
    if d == float("inf"):
        return float("inf")
    return d + 1.0  # assistant travel distance + one final pick action


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
            return 0.0
        return _assistant_value_from_pos(env, a_pos, g)
    dr, dc = ACTION_DELTAS.get(assistant_action, (0, 0))
    r, c = a_pos
    nr, nc = r + dr, c + dc
    if (nr, nc) in env.walls or not (0 <= nr < env.N and 0 <= nc < env.N):
        new_a_pos = a_pos
    else:
        new_a_pos = (nr, nc)
    return _assistant_value_from_pos(env, new_a_pos, g)


def _normalized_question_value(env, state, posterior, candidate_goals, q, answer_noise):
    expected_V = 0.0
    total_mass = 0.0
    for ans in q[1]:
        p_ans = sum(
            posterior.get(g, 0.0) * answer_likelihood(q, ans, g, env, answer_noise)
            for g in candidate_goals
        )
        if p_ans <= 0:
            continue
        p_new = {
            g: posterior.get(g, 0.0) * answer_likelihood(q, ans, g, env, answer_noise)
            for g in candidate_goals
        }
        z = sum(p_new.values())
        if z <= 0:
            continue
        for g in candidate_goals:
            p_new[g] /= z
        g_star = max(candidate_goals, key=lambda g: p_new.get(g, 0.0))
        task_act = assistant_task_action(env, state, g_star)
        v_ans = sum(
            p_new.get(g, 0.0) * V_after_action_approx(env, state, g, task_act)
            for g in candidate_goals
        )
        expected_V += p_ans * v_ans
        total_mass += p_ans
    if total_mass <= 0:
        return float("inf")
    return expected_V / total_mass


def _posterior_entropy_norm(posterior, candidate_goals):
    """Entropy of posterior restricted to candidate_goals, with normalization safety."""
    masses = [max(0.0, posterior.get(g, 0.0)) for g in candidate_goals]
    z = sum(masses)
    if z <= 0:
        return 0.0
    ent = 0.0
    for m in masses:
        if m <= 0:
            continue
        p = m / z
        ent -= p * math.log(p)
    return ent


def question_info_gain(env, posterior, candidate_goals, q, answer_noise):
    """
    Information gain:
      IG(q) = H(b) - E_a[H(b_{q,a})]
    where b is posterior over goals and a is answer.
    """
    h_before = _posterior_entropy_norm(posterior, candidate_goals)
    expected_h = 0.0
    total_mass = 0.0
    for ans in q[1]:
        p_ans = sum(
            posterior.get(g, 0.0) * answer_likelihood(q, ans, g, env, answer_noise)
            for g in candidate_goals
        )
        if p_ans <= 0:
            continue
        p_new = {
            g: posterior.get(g, 0.0) * answer_likelihood(q, ans, g, env, answer_noise)
            for g in candidate_goals
        }
        h_post = _posterior_entropy_norm(p_new, candidate_goals)
        expected_h += p_ans * h_post
        total_mass += p_ans
    if total_mass > 0:
        expected_h /= total_mass
    else:
        expected_h = h_before
    ig = h_before - expected_h
    return ig


def best_question_info_gain(env, posterior, candidate_goals, answer_noise, asked_qnames=None):
    """Return (best_qname, best_ig) over unasked questions."""
    asked = set(asked_qnames or [])
    best_q = None
    best_ig = -float("inf")
    for q in list_questions():
        if q[0] in asked:
            continue
        ig = question_info_gain(env, posterior, candidate_goals, q, answer_noise)
        if ig > best_ig:
            best_ig = ig
            best_q = q[0]
    if best_q is None:
        return None, -float("inf")
    return best_q, best_ig


def best_question_cost(state, posterior, candidate_goals, env, answer_noise, question_cost=None, asked_qnames=None):
    question_cost = question_cost or config.QUESTION_COST
    asked = set(asked_qnames or [])
    best_q = None
    best_cost = float("inf")
    for q in list_questions():
        if q[0] in asked:
            continue
        expected_v = _normalized_question_value(env, state, posterior, candidate_goals, q, answer_noise)
        if expected_v == float("inf"):
            continue
        cost_q = 1.0 + question_cost + expected_v
        if cost_q < best_cost:
            best_cost = cost_q
            best_q = q[0]
    return best_q, best_cost


def CostAct(state, posterior, candidate_goals, env, answer_noise):
    if not candidate_goals or not posterior:
        return 1.0
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    task_act = assistant_task_action(env, state, g_star)
    V = sum(posterior.get(g, 0) * V_after_action_approx(env, state, g, task_act) for g in candidate_goals if posterior.get(g, 0) > 0)
    return 1.0 + V


def CostAsk(state, posterior, candidate_goals, env, answer_noise, question_cost=None, asked_qnames=None):
    if not candidate_goals or not posterior:
        return 1.0 + (question_cost or config.QUESTION_COST)
    _, best_cost = best_question_cost(
        state=state,
        posterior=posterior,
        candidate_goals=candidate_goals,
        env=env,
        answer_noise=answer_noise,
        question_cost=question_cost,
        asked_qnames=asked_qnames,
    )
    return best_cost


def policy_never_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                    rng, beta, eps, answer_noise, **kwargs):
    if not candidate_goals or not posterior:
        return "stay", posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior


def policy_always_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                      questions_asked, rng, beta, eps, answer_noise, max_questions=None,
                      entropy_threshold=None, asked_qnames=None, **kwargs):
    max_questions = max_questions or config.MAX_QUESTIONS
    entropy_threshold = entropy_threshold or config.ENTROPY_THRESHOLD
    if not candidate_goals or not posterior:
        return "stay", posterior
    ent = posterior_entropy(posterior)
    if questions_asked < max_questions and ent > entropy_threshold:
        best_q, _ = best_question_cost(
            state=state,
            posterior=posterior,
            candidate_goals=candidate_goals,
            env=env,
            answer_noise=answer_noise,
            question_cost=config.QUESTION_COST,
            asked_qnames=asked_qnames,
        )
        if best_q is not None:
            return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior


def policy_ask_or_act(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                      questions_asked, rng, beta, eps, answer_noise, question_cost=None,
                      max_questions=None, entropy_gate=None, ask_window=None, asked_qnames=None, **kwargs):
    question_cost = question_cost or config.QUESTION_COST
    max_questions = max_questions or config.MAX_QUESTIONS
    entropy_gate = entropy_gate if entropy_gate is not None else config.ENTROPY_GATE
    ask_window = ask_window if ask_window is not None else config.ASK_WINDOW
    if not candidate_goals or not posterior:
        return "stay", posterior
    if questions_asked >= max_questions:
        g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
        return assistant_task_action(env, state, g_star), posterior
    step_t = len(principal_action_history)
    if step_t > ask_window:
        g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
        return assistant_task_action(env, state, g_star), posterior
    ent = posterior_entropy(posterior)
    if ent <= entropy_gate:
        g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
        return assistant_task_action(env, state, g_star), posterior

    cost_act = CostAct(state, posterior, candidate_goals, env, answer_noise)
    best_q, cost_ask = best_question_cost(
        state=state,
        posterior=posterior,
        candidate_goals=candidate_goals,
        env=env,
        answer_noise=answer_noise,
        question_cost=question_cost,
        asked_qnames=asked_qnames,
    )
    if best_q is not None and cost_ask < cost_act:
        return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0))
    return assistant_task_action(env, state, g_star), posterior


def policy_info_gain_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                         questions_asked, rng, beta, eps, answer_noise, max_questions=None,
                         entropy_threshold=None, asked_qnames=None, ig_threshold=None,
                         infogain_use_entropy_gate=None, **kwargs):
    """
    Ask the highest information-gain question if uncertainty is high enough.
    Otherwise, act toward MAP goal.
    """
    if not candidate_goals or not posterior:
        return "stay", posterior
    max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
    entropy_threshold = config.ENTROPY_THRESHOLD if entropy_threshold is None else entropy_threshold
    ig_threshold = config.IG_THRESHOLD if ig_threshold is None else ig_threshold
    use_entropy_gate = (
        config.INFOGAIN_USE_ENTROPY_GATE
        if infogain_use_entropy_gate is None
        else infogain_use_entropy_gate
    )
    if questions_asked < max_questions:
        best_q, best_ig = best_question_info_gain(
            env=env,
            posterior=posterior,
            candidate_goals=candidate_goals,
            answer_noise=answer_noise,
            asked_qnames=asked_qnames,
        )
        ent = posterior_entropy(posterior)
        entropy_ok = (not use_entropy_gate) or (ent > entropy_threshold)
        if best_q is not None and best_ig > ig_threshold and entropy_ok:
            return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0.0))
    return assistant_task_action(env, state, g_star), posterior


def policy_easy_info_gain_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                              questions_asked, rng, beta, eps, answer_noise, max_questions=None,
                              asked_qnames=None, ig_threshold=None, **kwargs):
    """
    Easy IG baseline:
      - choose q* = argmax IG(q) over available questions
      - ask iff IG(q*) >= IG_THRESHOLD and question budget remains
      - otherwise stop asking and act on MAP goal
    """
    if not candidate_goals or not posterior:
        return "stay", posterior
    max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
    max_questions = min(max_questions, int(getattr(config, "EASY_IG_MAX_QUESTIONS", 1)))
    ig_threshold = config.IG_THRESHOLD if ig_threshold is None else ig_threshold
    can_ask = (max_questions is None) or (questions_asked < max_questions)
    if can_ask:
        best_q, best_ig = best_question_info_gain(
            env=env,
            posterior=posterior,
            candidate_goals=candidate_goals,
            answer_noise=answer_noise,
            asked_qnames=asked_qnames,
        )
        if best_q is not None and best_ig >= ig_threshold:
            return ("ask", best_q), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0.0))
    return assistant_task_action(env, state, g_star), posterior


def policy_random_ask(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                      questions_asked, rng, beta, eps, answer_noise, max_questions=None,
                      entropy_threshold=None, asked_qnames=None, ig_threshold=None,
                      infogain_use_entropy_gate=None, **kwargs):
    """
    Same ask gating as info_gain_ask, but choose a random available question.
    RNG is episode-seeded by caller, so behavior is deterministic for a fixed seed.
    """
    if not candidate_goals or not posterior:
        return "stay", posterior
    max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
    entropy_threshold = config.ENTROPY_THRESHOLD if entropy_threshold is None else entropy_threshold
    ig_threshold = config.IG_THRESHOLD if ig_threshold is None else ig_threshold
    use_entropy_gate = (
        config.INFOGAIN_USE_ENTROPY_GATE
        if infogain_use_entropy_gate is None
        else infogain_use_entropy_gate
    )
    if questions_asked < max_questions:
        _, best_ig = best_question_info_gain(
            env=env,
            posterior=posterior,
            candidate_goals=candidate_goals,
            answer_noise=answer_noise,
            asked_qnames=asked_qnames,
        )
        ent = posterior_entropy(posterior)
        entropy_ok = (not use_entropy_gate) or (ent > entropy_threshold)
        if best_ig > ig_threshold and entropy_ok:
            asked = set(asked_qnames or [])
            qnames = [q[0] for q in list_questions() if q[0] not in asked]
            if qnames:
                q_idx = int(rng.integers(0, len(qnames)))
                return ("ask", qnames[q_idx]), posterior
    g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0.0))
    return assistant_task_action(env, state, g_star), posterior


def _normalize_posterior_for_goals(posterior, candidate_goals):
    if not candidate_goals:
        return {}
    p = {g: max(0.0, float(posterior.get(g, 0.0))) for g in candidate_goals}
    z = sum(p.values())
    if z > 0:
        return {g: p[g] / z for g in candidate_goals}
    u = 1.0 / len(candidate_goals)
    return {g: u for g in candidate_goals}


def _sample_goal_from_belief(rng, posterior, candidate_goals):
    p = _normalize_posterior_for_goals(posterior, candidate_goals)
    goals = list(candidate_goals)
    probs = [p[g] for g in goals]
    if not goals:
        return None
    return rng.choice(goals, p=probs)


def _move_with_walls(pos, action, N, walls):
    if action == "pick":
        return pos
    dr, dc = ACTION_DELTAS.get(action, (0, 0))
    r, c = pos
    nr, nc = r + dr, c + dc
    if (nr, nc) in walls or not (0 <= nr < N and 0 <= nc < N):
        return pos
    return (nr, nc)


def _principal_actions():
    if getattr(config, "PRINCIPAL_CAN_PICK", True):
        return ACTIONS
    return tuple(a for a in ACTIONS if a != "pick")


def _principal_action_probs_sim(principal_pos, goal_pos, N, walls, beta, eps):
    actions = _principal_actions()
    if goal_pos is None:
        return {a: 1.0 / len(actions) for a in actions}
    scores = {}
    finite_dists = []
    for a in actions:
        if a in ("stay", "pick"):
            next_pos = principal_pos
        else:
            next_pos = _move_with_walls(principal_pos, a, N, walls)
        d = grid_distance(N, walls, next_pos, goal_pos)
        if math.isfinite(d):
            finite_dists.append(d)
            scores[a] = -beta * d
        else:
            scores[a] = -1e6
    if not finite_dists:
        return {a: 1.0 / len(actions) for a in actions}
    max_s = max(scores.values())
    probs = {a: math.exp(scores[a] - max_s) for a in actions}
    z = sum(probs.values())
    probs = {a: (1 - eps) * (probs[a] / z) + eps / len(actions) for a in actions}
    return probs


def _sample_principal_action_sim(rng, principal_pos, goal_pos, N, walls, beta, eps):
    probs = _principal_action_probs_sim(principal_pos, goal_pos, N, walls, beta, eps)
    actions = list(probs.keys())
    p = [probs[a] for a in actions]
    return rng.choice(actions, p=p)


def _object_at_pos(pos, collected, pos_to_obj_ids, obj_idx):
    for obj_id in pos_to_obj_ids.get(pos, []):
        if not collected[obj_idx[obj_id]]:
            return obj_id
    return None


def _update_posterior_from_principal_sim(posterior, principal_action, principal_pos, candidate_goals,
                                          collected, obj_pos, obj_idx, N, walls, beta, eps):
    if not candidate_goals:
        return {}
    post = _normalize_posterior_for_goals(posterior, candidate_goals)
    log_post = {}
    for g in candidate_goals:
        prev = post.get(g, 0.0)
        if prev <= 0:
            continue
        g_pos = None if collected[obj_idx[g]] else obj_pos[g]
        probs = _principal_action_probs_sim(principal_pos, g_pos, N, walls, beta, eps)
        p_a = max(1e-12, float(probs.get(principal_action, 0.0)))
        log_post[g] = math.log(prev + 1e-15) + math.log(p_a)
    if not log_post:
        return _normalize_posterior_for_goals(post, candidate_goals)
    m = max(log_post.values())
    z = sum(math.exp(v - m) for v in log_post.values())
    out = {g: (math.exp(log_post[g] - m) / z if g in log_post else 0.0) for g in candidate_goals}
    return out


def _heuristic_remaining_cost(assistant_pos, hidden_goal, collected, obj_idx, obj_pos, N, walls, sim_steps, episode_max_steps):
    if hidden_goal is None:
        return 0.0
    if collected[obj_idx[hidden_goal]]:
        return 0.0
    d = grid_distance(N, walls, assistant_pos, obj_pos[hidden_goal])
    if d == float("inf"):
        return float(episode_max_steps)
    rem = float(d + 1)
    if sim_steps + rem > episode_max_steps:
        return float(episode_max_steps)
    return rem


def _posterior_key(posterior, candidate_goals):
    p = _normalize_posterior_for_goals(posterior, candidate_goals)
    return tuple(round(float(p.get(g, 0.0)), 3) for g in candidate_goals)


def _sim_transition(env, rng, action, hidden_goal, posterior, candidate_goals, principal_pos, assistant_pos, collected,
                    q_count, q_mask, sim_steps, question_cost, ask_counts_as_step, wrong_pick_fail,
                    principal_beta, principal_eps, assistant_beta, assistant_eps, answer_noise, episode_max_steps,
                    qnames, q_to_tuple, q_to_bit, obj_pos, obj_idx, pos_to_obj_ids):
    N, walls = env.N, env.walls
    candidate_goals = tuple(candidate_goals)
    posterior = _normalize_posterior_for_goals(posterior, candidate_goals)

    if isinstance(action, tuple) and action[0] == "ask":
        qname = action[1]
        q_tuple = q_to_tuple.get(qname)
        if q_tuple is None:
            return posterior, candidate_goals, principal_pos, assistant_pos, collected, q_count, q_mask, sim_steps, False, False, 0.0
        answer_space = list(q_tuple[1])
        p_ans = [max(0.0, float(answer_likelihood(q_tuple, a, hidden_goal, env, answer_noise))) for a in answer_space]
        z = sum(p_ans)
        if z <= 0:
            p_ans = [1.0 / len(answer_space) for _ in answer_space]
        else:
            p_ans = [x / z for x in p_ans]
        ans = rng.choice(answer_space, p=p_ans)
        p_new = {
            g: posterior.get(g, 0.0) * answer_likelihood(q_tuple, ans, g, env, answer_noise)
            for g in candidate_goals
        }
        posterior = _normalize_posterior_for_goals(p_new, candidate_goals)
        q_count2 = q_count + 1
        q_mask2 = q_mask | q_to_bit.get(qname, 0)
        step_inc = 1 if ask_counts_as_step else 0
        sim_steps2 = sim_steps + step_inc
        done = sim_steps2 >= episode_max_steps
        cost = float(step_inc + question_cost)
        return posterior, candidate_goals, principal_pos, assistant_pos, collected, q_count2, q_mask2, sim_steps2, False, done, cost

    # Physical action: simulate principal action then assistant action.
    goal_pos = None if hidden_goal is None or collected[obj_idx[hidden_goal]] else obj_pos[hidden_goal]
    principal_action = _sample_principal_action_sim(
        rng, principal_pos, goal_pos, N, walls, principal_beta, principal_eps
    )
    posterior = _update_posterior_from_principal_sim(
        posterior, principal_action, principal_pos, candidate_goals,
        collected, obj_pos, obj_idx, N, walls, assistant_beta, assistant_eps
    )
    p_eff_action = principal_action
    if principal_action == "pick" and not getattr(config, "PRINCIPAL_CAN_PICK", True):
        p_eff_action = "stay"
    principal_pos2 = _move_with_walls(principal_pos, p_eff_action, N, walls)
    collected2 = collected
    if p_eff_action == "pick":
        p_obj = _object_at_pos(principal_pos2, collected2, pos_to_obj_ids, obj_idx)
        if p_obj is not None:
            idx = obj_idx[p_obj]
            collected2 = tuple((collected2[i] or (i == idx)) for i in range(len(collected2)))

    assistant_pos2 = _move_with_walls(assistant_pos, action, N, walls)
    success = False
    done = False
    candidate_goals2 = tuple(candidate_goals)
    posterior2 = dict(posterior)
    if action == "pick":
        picked = _object_at_pos(assistant_pos2, collected2, pos_to_obj_ids, obj_idx)
        if picked is not None:
            idx = obj_idx[picked]
            collected2 = tuple((collected2[i] or (i == idx)) for i in range(len(collected2)))
            if hidden_goal is not None and picked == hidden_goal:
                success = True
                done = True
            elif hidden_goal is not None and wrong_pick_fail:
                done = True
            elif hidden_goal is not None:
                # Recoverable wrong pick: eliminate hypothesis, mirroring eval loop behavior.
                if picked in candidate_goals2:
                    posterior2[picked] = 0.0
                    candidate_goals2 = tuple(g for g in candidate_goals2 if g != picked)
                    posterior2 = _normalize_posterior_for_goals(posterior2, candidate_goals2)
    sim_steps2 = sim_steps + 1
    if (not success) and sim_steps2 >= episode_max_steps:
        done = True
    cost = 1.0
    return posterior2, candidate_goals2, principal_pos2, assistant_pos2, collected2, q_count, q_mask, sim_steps2, success, done, cost


def policy_pomcp_planner(env, state, instruction_u, posterior, candidate_goals, principal_action_history,
                         questions_asked, rng, beta, eps, answer_noise, question_cost=None,
                         max_questions=None, asked_qnames=None, current_steps=None,
                         episode_max_steps=None, ask_counts_as_step=None, wrong_pick_fail=None, **kwargs):
    """
    POMDP-style Monte Carlo planning baseline (POMCP/POUCT approximation).
    Actions include physical controls and ask actions; hidden goal is sampled from belief particles.
    """
    if not candidate_goals or not posterior:
        return "stay", posterior

    question_cost = config.QUESTION_COST if question_cost is None else question_cost
    max_questions = config.MAX_QUESTIONS if max_questions is None else max_questions
    ask_counts_as_step = config.ASK_COUNTS_AS_STEP if ask_counts_as_step is None else ask_counts_as_step
    wrong_pick_fail = config.WRONG_PICK_FAIL if wrong_pick_fail is None else wrong_pick_fail
    current_steps = 0 if current_steps is None else int(current_steps)
    episode_max_steps = config.MAX_STEPS if episode_max_steps is None else int(episode_max_steps)
    principal_beta = float(kwargs.get("principal_beta", beta))
    principal_eps = float(kwargs.get("principal_eps", eps))
    assistant_beta = float(kwargs.get("assistant_beta", beta))
    assistant_eps = float(kwargs.get("assistant_eps", eps))
    min_k = int(getattr(config, "POMCP_MIN_K", 2))

    remaining_deadline = max(1, episode_max_steps - current_steps)
    horizon_cfg = kwargs.get("pomcp_horizon", None)
    n_iters_cfg = kwargs.get("pomcp_iters", None)
    uct_cfg = kwargs.get("pomcp_uct_c", None)
    horizon = min(
        int(getattr(config, "POMCP_HORIZON", 8) if horizon_cfg is None else horizon_cfg),
        remaining_deadline,
    )
    n_iters = int(getattr(config, "POMCP_ITERS", 250) if n_iters_cfg is None else n_iters_cfg)
    uct_c = float(getattr(config, "POMCP_UCT_C", 1.4) if uct_cfg is None else uct_cfg)

    # Runtime guard: low ambiguity or near-deadline falls back to greedy MAP.
    if len(candidate_goals) < min_k or remaining_deadline <= 1:
        g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0.0))
        return assistant_task_action(env, state, g_star), posterior

    q_list = list_questions()
    qnames = [q[0] for q in q_list]
    q_to_tuple = {q[0]: q for q in q_list}
    q_to_bit = {q[0]: (1 << i) for i, q in enumerate(q_list)}
    q_mask0 = 0
    for qn in (asked_qnames or []):
        q_mask0 |= q_to_bit.get(qn, 0)

    obj_ids = [o["obj_id"] for o in env.objects]
    obj_idx = {g: i for i, g in enumerate(obj_ids)}
    obj_pos = {o["obj_id"]: o["pos"] for o in env.objects}
    pos_to_obj_ids = {}
    for o in env.objects:
        pos_to_obj_ids.setdefault(o["pos"], []).append(o["obj_id"])
    collected0 = tuple(bool(o.get("collected", False)) for o in env.objects)

    principal_pos0 = state["principal_pos"]
    assistant_pos0 = state["assistant_pos"]
    candidate_goals0 = tuple(candidate_goals)
    posterior0 = _normalize_posterior_for_goals(posterior, candidate_goals0)

    def legal_actions(q_count, q_mask):
        acts = list(ACTIONS)
        can_ask = (max_questions is None) or (q_count < max_questions)
        if can_ask:
            for qn in qnames:
                if (q_mask & q_to_bit[qn]) == 0:
                    acts.append(("ask", qn))
        return acts

    def state_key(principal_pos, assistant_pos, collected, cand_goals, post, q_count, q_mask, sim_steps, depth):
        return (
            principal_pos,
            assistant_pos,
            collected,
            tuple(cand_goals),
            q_count,
            q_mask,
            sim_steps,
            depth,
            _posterior_key(post, cand_goals),
        )

    tree = {}

    def rollout(hidden_goal, post, cand_goals, p_pos, a_pos, coll, q_count, q_mask, s_steps, depth):
        if hidden_goal is None:
            return 0.0
        if depth <= 0:
            return -_heuristic_remaining_cost(a_pos, hidden_goal, coll, obj_idx, obj_pos, env.N, env.walls, s_steps, episode_max_steps)
        if s_steps >= episode_max_steps:
            return -float(episode_max_steps)
        if hidden_goal in obj_idx and coll[obj_idx[hidden_goal]]:
            return 0.0
        if _object_at_pos(a_pos, coll, pos_to_obj_ids, obj_idx) is not None:
            a = "pick"
        else:
            gpos = obj_pos.get(hidden_goal, a_pos)
            a = _bfs_next_step(env.N, env.walls, a_pos, gpos)
        out = _sim_transition(
            env, rng, a, hidden_goal, post, cand_goals, p_pos, a_pos, coll, q_count, q_mask, s_steps,
            question_cost, ask_counts_as_step, wrong_pick_fail,
            principal_beta, principal_eps, assistant_beta, assistant_eps, answer_noise, episode_max_steps,
            qnames, q_to_tuple, q_to_bit, obj_pos, obj_idx, pos_to_obj_ids,
        )
        (post2, cand2, p2, a2, coll2, q2, qm2, s2, success, done, c) = out
        if success:
            return -c
        if done:
            return -(c + float(episode_max_steps))
        return -c + rollout(hidden_goal, post2, cand2, p2, a2, coll2, q2, qm2, s2, depth - 1)

    def simulate(hidden_goal, post, cand_goals, p_pos, a_pos, coll, q_count, q_mask, s_steps, depth):
        if hidden_goal is None:
            return 0.0
        if depth <= 0:
            return -_heuristic_remaining_cost(a_pos, hidden_goal, coll, obj_idx, obj_pos, env.N, env.walls, s_steps, episode_max_steps)
        if s_steps >= episode_max_steps:
            return -float(episode_max_steps)
        if hidden_goal in obj_idx and coll[obj_idx[hidden_goal]]:
            return 0.0
        key = state_key(p_pos, a_pos, coll, cand_goals, post, q_count, q_mask, s_steps, depth)
        node = tree.setdefault(key, {"N": 0, "A": {}})
        acts = legal_actions(q_count, q_mask)
        if not acts:
            return -_heuristic_remaining_cost(a_pos, hidden_goal, coll, obj_idx, obj_pos, env.N, env.walls, s_steps, episode_max_steps)
        for a in acts:
            node["A"].setdefault(a, [0, 0.0])  # [N, W]
        untried = [a for a in acts if node["A"][a][0] == 0]
        if untried:
            a = untried[int(rng.integers(0, len(untried)))]
            out = _sim_transition(
                env, rng, a, hidden_goal, post, cand_goals, p_pos, a_pos, coll, q_count, q_mask, s_steps,
                question_cost, ask_counts_as_step, wrong_pick_fail,
                principal_beta, principal_eps, assistant_beta, assistant_eps, answer_noise, episode_max_steps,
                qnames, q_to_tuple, q_to_bit, obj_pos, obj_idx, pos_to_obj_ids,
            )
            (post2, cand2, p2, a2, coll2, q2, qm2, s2, success, done, c) = out
            if success:
                ret = -c
            elif done:
                ret = -(c + float(episode_max_steps))
            else:
                ret = -c + rollout(hidden_goal, post2, cand2, p2, a2, coll2, q2, qm2, s2, depth - 1)
            node["N"] += 1
            node["A"][a][0] += 1
            node["A"][a][1] += ret
            return ret

        # UCT action selection.
        ln_n = math.log(max(1, node["N"]))
        best_a = None
        best_score = -float("inf")
        for a in acts:
            n_a, w_a = node["A"][a]
            q_a = w_a / max(1, n_a)
            ucb = q_a + uct_c * math.sqrt(ln_n / max(1, n_a))
            if ucb > best_score:
                best_score = ucb
                best_a = a

        out = _sim_transition(
            env, rng, best_a, hidden_goal, post, cand_goals, p_pos, a_pos, coll, q_count, q_mask, s_steps,
            question_cost, ask_counts_as_step, wrong_pick_fail,
            principal_beta, principal_eps, assistant_beta, assistant_eps, answer_noise, episode_max_steps,
            qnames, q_to_tuple, q_to_bit, obj_pos, obj_idx, pos_to_obj_ids,
        )
        (post2, cand2, p2, a2, coll2, q2, qm2, s2, success, done, c) = out
        if success:
            ret = -c
        elif done:
            ret = -(c + float(episode_max_steps))
        else:
            ret = -c + simulate(hidden_goal, post2, cand2, p2, a2, coll2, q2, qm2, s2, depth - 1)
        node["N"] += 1
        node["A"][best_a][0] += 1
        node["A"][best_a][1] += ret
        return ret

    root_key = state_key(
        principal_pos0, assistant_pos0, collected0, candidate_goals0, posterior0,
        questions_asked, q_mask0, current_steps, horizon
    )
    tree.setdefault(root_key, {"N": 0, "A": {}})

    for _ in range(max(1, n_iters)):
        g_hidden = _sample_goal_from_belief(rng, posterior0, candidate_goals0)
        simulate(
            g_hidden, posterior0, candidate_goals0, principal_pos0, assistant_pos0, collected0,
            questions_asked, q_mask0, current_steps, horizon
        )

    root = tree.get(root_key)
    if not root or not root["A"]:
        g_star = max(candidate_goals0, key=lambda g: posterior0.get(g, 0.0))
        return assistant_task_action(env, state, g_star), posterior

    best_action = None
    best_q = -float("inf")
    for a, (n_a, w_a) in root["A"].items():
        if n_a <= 0:
            continue
        q_a = w_a / n_a
        if q_a > best_q:
            best_q = q_a
            best_action = a

    if best_action is None:
        g_star = max(candidate_goals0, key=lambda g: posterior0.get(g, 0.0))
        return assistant_task_action(env, state, g_star), posterior
    return best_action, posterior
