"""Evaluation: run_episode, run_sweep, run_ablations. Optional parallel workers."""

import csv
import json
import os
import numpy as np
from src import config
from src.env import grid_distance
from src.world import generate_world, instruction_to_candidate_goals, answer_question, answer_likelihood, list_questions
from src.world.instructions import TEMPLATES, get_template_id_for_ambiguity
from src.inference import init_posterior, update_posterior, posterior_entropy
from src.agents import (
    sample_principal_action,
    policy_ask_or_act,
    policy_never_ask,
    policy_always_ask,
    policy_info_gain_ask,
    policy_easy_info_gain_ask,
    policy_random_ask,
    policy_pomcp_planner,
    _bfs_next_step,
)
from src.agents.assistant import CostAct, best_question_cost, assistant_task_action, question_info_gain


def episode_optimal_steps(env, true_goal_obj_id):
    """Assistant-only shortest steps to finish = shortest path to true goal + pick."""
    state = env.get_state()
    obj = env.get_object_by_id(true_goal_obj_id)
    if obj is None or obj.get("collected", False):
        return 1
    d = grid_distance(env.N, env.walls, state["assistant_pos"], obj["pos"])
    if d == float("inf"):
        return config.MAX_STEPS
    return int(d + 1)


def oracle_steps(env, true_goal_obj_id, rng):
    """Oracle assistant: shortest path to true goal then pick. Returns steps (path + 1 for pick)."""
    env.reset(rng.integers(0, 2**31))
    state = env.get_state()
    steps = 0
    obj = env.get_object_by_id(true_goal_obj_id)
    if obj is None or obj.get("collected", False):
        return 0
    goal_pos = obj["pos"]
    while steps < config.MAX_STEPS:
        a_pos = state["assistant_pos"]
        if a_pos == goal_pos:
            env.step("stay", "pick", true_goal_obj_id=true_goal_obj_id)
            steps += 1
            return steps
        act = _bfs_next_step(env.N, env.walls, a_pos, goal_pos)
        env.step("stay", act, true_goal_obj_id=true_goal_obj_id)
        steps += 1
        state = env.get_state()
    return steps


def _episode_question_cap(max_questions_per_episode):
    if max_questions_per_episode in (None, -1):
        return None
    return max(0, int(max_questions_per_episode))


def _compute_team_cost(steps, questions_asked, success, episode_max_steps, question_cost):
    """
    Team cost used for regret:
      success: steps + QUESTION_COST * questions_asked
      failure: episode_max_steps + QUESTION_COST * questions_asked
    """
    base_steps = steps if success else episode_max_steps
    return float(base_steps + question_cost * questions_asked)


def _normalized_goal_probs(posterior, candidate_goals):
    if not candidate_goals:
        return []
    probs = [max(0.0, float(posterior.get(g, 0.0))) for g in candidate_goals]
    z = sum(probs)
    if z > 0:
        return [p / z for p in probs]
    u = 1.0 / len(candidate_goals)
    return [u for _ in candidate_goals]


def _effective_goal_count(posterior, candidate_goals):
    probs = _normalized_goal_probs(posterior, candidate_goals)
    if not probs:
        return 0.0
    denom = sum(p * p for p in probs)
    if denom <= 0:
        return 0.0
    return float(1.0 / denom)


def _clarification_snapshot(posterior, candidate_goals):
    return float(posterior_entropy(posterior)), _effective_goal_count(posterior, candidate_goals)


def _sweep_env_seed(base_seed, rep_seed, K, eps, beta, episode_id):
    """
    Stable deterministic seed for reproducible replicate evaluation:
      env_seed = base_seed + 100000*rep + 1000*K + 100*int(eps*100) + 10*beta + episode_id
    """
    return int(
        base_seed
        + 100000 * int(rep_seed)
        + 1000 * int(K)
        + 100 * int(eps * 100)
        + int(10 * float(beta))
        + int(episode_id)
    )


def _run_episode_impl(policy_name, seed, config_dict):
    """Single episode; used by run_episode, sweeps, and ablations."""
    K = config_dict.get("ambiguity_K", 2)
    eps = config_dict.get("eps", config.DEFAULT_EPS)
    beta = config_dict.get("beta", config.DEFAULT_BETA)
    principal_eps = config_dict.get("principal_eps", eps)
    assistant_eps = config_dict.get("assistant_eps", eps)
    principal_beta = config_dict.get("principal_beta", beta)
    assistant_beta = config_dict.get("assistant_beta", beta)
    answer_noise = config_dict.get("answer_noise", config.ANSWER_NOISE)
    N = config_dict.get("N", config.DEFAULT_N)
    M = config_dict.get("M", config.DEFAULT_M)
    allowed_template_ids = config_dict.get("allowed_template_ids", None)

    question_cost = config_dict.get("question_cost", config.QUESTION_COST)
    ask_counts_as_step = config_dict.get("ask_counts_as_step", config.ASK_COUNTS_AS_STEP)
    wrong_pick_fail = config_dict.get("wrong_pick_fail", config.WRONG_PICK_FAIL)
    max_questions_per_episode = config_dict.get("max_questions_per_episode", config.MAX_QUESTIONS_PER_EPISODE)
    use_dynamic_deadline = config_dict.get("use_dynamic_deadline", config.USE_DYNAMIC_DEADLINE)
    deadline_margin = config_dict.get("deadline_margin", config.DEADLINE_MARGIN)

    debug = config_dict.get("debug", config.DEBUG)
    debug_max_steps = config_dict.get("debug_max_steps", config.DEBUG_MAX_STEPS)
    debug_min_k = config_dict.get("debug_min_k", config.DEBUG_MIN_K)
    rng = np.random.default_rng(seed)

    env, instruction_u, true_goal_obj_id = generate_world(
        seed, N=N, M=M, ambiguity_K=K, allowed_template_ids=allowed_template_ids
    )
    template_id = get_template_id_for_ambiguity(instruction_u)
    candidate_goals = instruction_to_candidate_goals(instruction_u, env)

    optimal_steps = episode_optimal_steps(env, true_goal_obj_id)
    if use_dynamic_deadline:
        episode_max_steps = min(config.MAX_STEPS, optimal_steps + int(deadline_margin))
    else:
        episode_max_steps = config.MAX_STEPS

    if not candidate_goals:
        team_cost = _compute_team_cost(episode_max_steps, 0, False, episode_max_steps, question_cost)
        return {
            "success": False,
            "steps": episode_max_steps,
            "questions_asked": 0,
            "assistant_picked_goal": False,
            "final_map_correct": False,
            "regret": team_cost - optimal_steps,
            "oracle_steps": optimal_steps,
            "episode_max_steps": episode_max_steps,
            "team_cost": team_cost,
            "oracle_cost": float(optimal_steps),
            "terminated_by_wrong_pick": False,
            "failure_by_wrong_pick": False,
            "failure_by_timeout": False,
            "entropy_before_first_ask": 0.0,
            "entropy_after_first_ask": 0.0,
            "effective_goal_count_before": 0.0,
            "effective_goal_count_after": 0.0,
            "ig_of_first_asked_question": 0.0,
            "template_id": -1,
        }

    posterior = init_posterior(candidate_goals)
    principal_action_history = []
    questions_asked = 0
    asked_qnames = set()
    steps = 0
    success = False
    terminated_by_wrong_pick = False

    entropy_before_first_ask, effective_goal_count_before = _clarification_snapshot(posterior, candidate_goals)
    entropy_after_first_ask = entropy_before_first_ask
    effective_goal_count_after = effective_goal_count_before
    ig_of_first_asked_question = 0.0
    first_ask_recorded = False

    policy_fn = {
        "ask_or_act": policy_ask_or_act,
        "never_ask": policy_never_ask,
        "always_ask": policy_always_ask,
        "info_gain_ask": policy_info_gain_ask,
        "easy_info_gain_ask": policy_easy_info_gain_ask,
        "random_ask": policy_random_ask,
        "pomcp_planner": policy_pomcp_planner,
    }[policy_name]

    episode_q_cap = _episode_question_cap(max_questions_per_episode)
    policy_max_questions = config.MAX_QUESTIONS
    if episode_q_cap is not None:
        policy_max_questions = min(policy_max_questions, episode_q_cap)

    while steps < episode_max_steps:
        state = env.get_state()
        principal_action = sample_principal_action(
            state, true_goal_obj_id, env, rng, principal_beta, principal_eps
        )
        principal_action_history.append(principal_action)

        policy_kw = {
            "env": env,
            "state": state,
            "instruction_u": instruction_u,
            "posterior": posterior,
            "candidate_goals": candidate_goals,
            "principal_action_history": principal_action_history,
            "questions_asked": questions_asked,
            "rng": rng,
            "beta": assistant_beta,
            "eps": assistant_eps,
            "answer_noise": answer_noise,
            "question_cost": question_cost,
            "max_questions": policy_max_questions,
            "entropy_threshold": config.ENTROPY_THRESHOLD,
            "entropy_gate": config.ENTROPY_GATE,
            "ask_window": config.ASK_WINDOW,
            "ig_threshold": config.IG_THRESHOLD,
            "infogain_use_entropy_gate": config.INFOGAIN_USE_ENTROPY_GATE,
            "asked_qnames": asked_qnames,
            "principal_beta": principal_beta,
            "principal_eps": principal_eps,
            "assistant_beta": assistant_beta,
            "assistant_eps": assistant_eps,
            "current_steps": steps,
            "episode_max_steps": episode_max_steps,
            "ask_counts_as_step": ask_counts_as_step,
            "wrong_pick_fail": wrong_pick_fail,
            "pomcp_iters": config_dict.get("pomcp_iters", None),
            "pomcp_horizon": config_dict.get("pomcp_horizon", None),
            "pomcp_uct_c": config_dict.get("pomcp_uct_c", None),
        }

        debug_enabled = debug and K >= debug_min_k and steps < debug_max_steps and policy_name == "ask_or_act"
        if debug_enabled:
            ent = posterior_entropy(posterior)
            top_goals = sorted(candidate_goals, key=lambda g: posterior.get(g, 0.0), reverse=True)[:3]
            top_probs = [(g, round(float(posterior.get(g, 0.0)), 3)) for g in top_goals]
            dbg_cost_act = CostAct(state, posterior, candidate_goals, env, answer_noise)
            dbg_best_q, dbg_cost_ask = best_question_cost(
                state=state,
                posterior=posterior,
                candidate_goals=candidate_goals,
                env=env,
                answer_noise=answer_noise,
                question_cost=question_cost,
                asked_qnames=asked_qnames,
            )

        out, posterior = policy_fn(**policy_kw)

        if debug_enabled:
            cost_ask_str = "inf" if dbg_cost_ask == float("inf") else f"{dbg_cost_ask:.3f}"
            print(
                f"[DEBUG] t={steps + 1} u='{instruction_u}' H={ent:.3f} "
                f"top3={top_probs} cost_act={dbg_cost_act:.3f} "
                f"best_cost_ask={cost_ask_str} best_q={dbg_best_q} "
                f"chosen={out} deadline={episode_max_steps}"
            )

        if isinstance(out, tuple) and out[0] == "ask":
            if episode_q_cap is not None and questions_asked >= episode_q_cap:
                # Defensive fallback if a policy requests ask beyond hard episode cap.
                g_star = max(candidate_goals, key=lambda g: posterior.get(g, 0.0))
                out = assistant_task_action(env, state, g_star)
            else:
                _, q_id = out
                asked_qnames.add(q_id)
                questions_asked += 1
                if ask_counts_as_step:
                    steps += 1
                q_tuple = next((q for q in list_questions() if q[0] == q_id), None)
                if q_tuple is None:
                    continue
                if not first_ask_recorded:
                    entropy_before_first_ask, effective_goal_count_before = _clarification_snapshot(
                        posterior, candidate_goals
                    )
                    ig_of_first_asked_question = float(
                        question_info_gain(
                            env=env,
                            posterior=posterior,
                            candidate_goals=candidate_goals,
                            q=q_tuple,
                            answer_noise=answer_noise,
                        )
                    )
                ans = answer_question(q_tuple, true_goal_obj_id, env, rng, answer_noise)
                if debug_enabled:
                    print(f"[DEBUG] ask q={q_id} answer={ans}")
                p_new = {
                    g: posterior.get(g, 0.0) * answer_likelihood(q_tuple, ans, g, env, answer_noise)
                    for g in candidate_goals
                }
                z = sum(p_new.values())
                if z > 0:
                    for g in candidate_goals:
                        posterior[g] = p_new[g] / z
                if not first_ask_recorded:
                    entropy_after_first_ask, effective_goal_count_after = _clarification_snapshot(
                        posterior, candidate_goals
                    )
                    first_ask_recorded = True
                continue

        assistant_action = out
        update_posterior(
            posterior, state, principal_action, candidate_goals, env, assistant_beta, assistant_eps
        )
        state, done, info = env.step(
            principal_action,
            assistant_action,
            true_goal_obj_id=true_goal_obj_id,
            wrong_pick_fail=wrong_pick_fail,
        )
        steps += 1
        success = info.get("assistant_picked_goal", False)
        terminated_by_wrong_pick = info.get("terminated_by_wrong_pick", False)

        if assistant_action == "pick" and not success and not terminated_by_wrong_pick:
            picked_obj_id = info.get("picked_obj_id")
            if picked_obj_id is not None and picked_obj_id in candidate_goals:
                posterior[picked_obj_id] = 0.0
                candidate_goals = [g for g in candidate_goals if g != picked_obj_id]
                if candidate_goals:
                    z = sum(max(0.0, posterior.get(g, 0.0)) for g in candidate_goals)
                    if z > 0:
                        for g in candidate_goals:
                            posterior[g] = max(0.0, posterior.get(g, 0.0)) / z
                    else:
                        uniform = 1.0 / len(candidate_goals)
                        for g in candidate_goals:
                            posterior[g] = uniform
                else:
                    break

        if done or success:
            break

    final_map_correct = (
        max(candidate_goals, key=lambda g: posterior.get(g, 0.0)) == true_goal_obj_id
    ) if candidate_goals else False

    oracle_cost = float(optimal_steps)
    team_cost = _compute_team_cost(steps, questions_asked, success, episode_max_steps, question_cost)
    regret = float(team_cost - oracle_cost)
    failure_by_wrong_pick = (not success) and bool(terminated_by_wrong_pick)
    failure_by_timeout = (not success) and (steps >= episode_max_steps) and (not failure_by_wrong_pick)

    return {
        "success": success,
        "steps": steps,
        "questions_asked": questions_asked,
        "assistant_picked_goal": success,
        "final_map_correct": final_map_correct,
        "regret": regret,
        "oracle_steps": optimal_steps,
        "episode_max_steps": episode_max_steps,
        "team_cost": team_cost,
        "oracle_cost": oracle_cost,
        "terminated_by_wrong_pick": terminated_by_wrong_pick,
        "failure_by_wrong_pick": failure_by_wrong_pick,
        "failure_by_timeout": failure_by_timeout,
        "entropy_before_first_ask": float(entropy_before_first_ask),
        "entropy_after_first_ask": float(entropy_after_first_ask),
        "effective_goal_count_before": float(effective_goal_count_before),
        "effective_goal_count_after": float(effective_goal_count_after),
        "ig_of_first_asked_question": float(ig_of_first_asked_question),
        "template_id": int(template_id),
    }


def run_episode(policy_name, seed, config_dict):
    """Run one episode. config_dict can override config (ambiguity_K, eps, beta, etc.)."""
    return _run_episode_impl(policy_name, seed, config_dict)


def run_sweep(output_csv="results/metrics.csv", output_json="results/summary.json"):
    """Sweep over conditions; optionally use parallel workers. Writes CSV and JSON."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    rows = []
    n_workers = getattr(config, "N_WORKERS", 0)
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    total_per_condition = len(reps) * n_episodes_per_seed
    print(f"[sweep] episodes per (policy,K,eps,beta): {total_per_condition} ({len(reps)} reps x {n_episodes_per_seed} eps/rep)")

    if n_workers and n_workers > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = []
        for K in config.AMBIGUITY_LEVELS:
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in config.POLICIES:
                        for rep_seed in reps:
                            for episode_id in range(n_episodes_per_seed):
                                seed = _sweep_env_seed(config.BASE_SEED, rep_seed, K, eps, beta, episode_id)
                                tasks.append((policy_name, seed, cfg, rep_seed, episode_id))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_run_episode_impl, p, s, c): (p, s, c, rep_seed, episode_id)
                for p, s, c, rep_seed, episode_id in tasks
            }
            for fut in as_completed(futures):
                policy_name, seed, cfg, rep_seed, episode_id = futures[fut]
                try:
                    m = fut.result()
                except Exception:
                    m = {
                        "success": False,
                        "steps": config.MAX_STEPS,
                        "questions_asked": 0,
                        "assistant_picked_goal": False,
                        "final_map_correct": False,
                        "regret": float(config.MAX_STEPS),
                        "oracle_steps": 0,
                        "episode_max_steps": config.MAX_STEPS,
                        "team_cost": float(config.MAX_STEPS),
                        "oracle_cost": 0.0,
                        "terminated_by_wrong_pick": False,
                        "failure_by_wrong_pick": False,
                        "failure_by_timeout": False,
                        "entropy_before_first_ask": 0.0,
                        "entropy_after_first_ask": 0.0,
                        "effective_goal_count_before": 0.0,
                        "effective_goal_count_after": 0.0,
                        "ig_of_first_asked_question": 0.0,
                    }
                K, eps, beta = cfg["ambiguity_K"], cfg["eps"], cfg["beta"]
                rows.append({
                    "ambiguity_K": K,
                    "eps": eps,
                    "beta": beta,
                    "policy": policy_name,
                    "rep_seed": int(rep_seed),
                    "episode_id": int(episode_id),
                    "success": m["success"],
                    "steps": m["steps"],
                    "questions_asked": m["questions_asked"],
                    "assistant_picked_goal": m.get("assistant_picked_goal", m["success"]),
                    "final_map_correct": m.get("final_map_correct", False),
                    "oracle_steps": m.get("oracle_steps", 0),
                    "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                    "team_cost": m.get("team_cost", float(m["steps"])),
                    "oracle_cost": m.get("oracle_cost", float(m.get("oracle_steps", 0))),
                    "terminated_by_wrong_pick": m.get("terminated_by_wrong_pick", False),
                    "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                    "failure_by_timeout": m.get("failure_by_timeout", False),
                    "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                    "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                    "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                    "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                    "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                    "regret": m["regret"],
                })
    else:
        for K in config.AMBIGUITY_LEVELS:
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in config.POLICIES:
                        for rep_seed in reps:
                            for episode_id in range(n_episodes_per_seed):
                                seed = _sweep_env_seed(config.BASE_SEED, rep_seed, K, eps, beta, episode_id)
                                m = _run_episode_impl(policy_name, seed, cfg)
                                rows.append({
                                    "ambiguity_K": K,
                                    "eps": eps,
                                    "beta": beta,
                                    "policy": policy_name,
                                    "rep_seed": int(rep_seed),
                                    "episode_id": int(episode_id),
                                    "success": m["success"],
                                    "steps": m["steps"],
                                    "questions_asked": m["questions_asked"],
                                    "assistant_picked_goal": m.get("assistant_picked_goal", m["success"]),
                                    "final_map_correct": m.get("final_map_correct", False),
                                    "oracle_steps": m.get("oracle_steps", 0),
                                    "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                                    "team_cost": m.get("team_cost", float(m["steps"])),
                                    "oracle_cost": m.get("oracle_cost", float(m.get("oracle_steps", 0))),
                                    "terminated_by_wrong_pick": m.get("terminated_by_wrong_pick", False),
                                    "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                                    "failure_by_timeout": m.get("failure_by_timeout", False),
                                    "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                                    "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                                    "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                                    "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                                    "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                                    "regret": m["regret"],
                                })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ambiguity_K",
                "eps",
                "beta",
                "policy",
                "rep_seed",
                "episode_id",
                "success",
                "steps",
                "questions_asked",
                "assistant_picked_goal",
                "final_map_correct",
                "oracle_steps",
                "episode_max_steps",
                "team_cost",
                "oracle_cost",
                "terminated_by_wrong_pick",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
                "regret",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for K in config.AMBIGUITY_LEVELS:
        for eps in config.EPS_LEVELS:
            for beta in config.BETA_LEVELS:
                key = (K, eps, beta)
                summary[key] = {}
                for policy_name in config.POLICIES:
                    sub = [
                        r for r in rows
                        if r["ambiguity_K"] == K and r["eps"] == eps and r["beta"] == beta and r["policy"] == policy_name
                    ]
                    summary[key][policy_name] = {
                        "success_rate": np.mean([r["success"] for r in sub]),
                        "avg_steps": np.mean([r["steps"] for r in sub]),
                        "avg_questions": np.mean([r["questions_asked"] for r in sub]),
                        "avg_regret": np.mean([r["regret"] for r in sub]),
                        "avg_map_correct": np.mean([r["final_map_correct"] for r in sub]),
                    }
    with open(output_json, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    return rows, summary


def run_ablations(output_csv="results/metrics_ablations.csv"):
    """Run fixed ablation grid and write a single metrics CSV."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)

    modes = [
        {"mode_name": "modeA", "question_cost": 0.0, "ask_counts_as_step": True},
        {"mode_name": "modeB", "question_cost": 0.5, "ask_counts_as_step": False},
        {"mode_name": "modeC", "question_cost": 0.5, "ask_counts_as_step": True},
    ]
    ks = [2, 3, 4]
    eps_levels = [0.0, 0.1]
    answer_noises = [0.0]
    wrong_pick_fail_options = [False, True]
    deadline_margins = [3, 5, 7]
    n_eps = 12

    rows = []
    for mode in modes:
        for wrong_pick_fail in wrong_pick_fail_options:
            for deadline_margin in deadline_margins:
                for K in ks:
                    for eps in eps_levels:
                        for answer_noise in answer_noises:
                            for policy in config.POLICIES:
                                for ep in range(n_eps):
                                    seed = (
                                        900000
                                        + K * 10000
                                        + int(eps * 100) * 100
                                        + (0 if mode["mode_name"] == "modeA" else 1 if mode["mode_name"] == "modeB" else 2) * 1000
                                        + (100 if wrong_pick_fail else 0)
                                        + deadline_margin * 10
                                        + config.POLICIES.index(policy) * 100
                                        + ep
                                    )
                                    cfg = {
                                        "ambiguity_K": K,
                                        "eps": eps,
                                        "beta": config.DEFAULT_BETA,
                                        "answer_noise": answer_noise,
                                        "question_cost": mode["question_cost"],
                                        "ask_counts_as_step": mode["ask_counts_as_step"],
                                        "wrong_pick_fail": wrong_pick_fail,
                                        "deadline_margin": deadline_margin,
                                    }
                                    m = _run_episode_impl(policy, seed, cfg)
                                    rows.append({
                                        "policy": policy,
                                        "K": K,
                                        "eps": eps,
                                        "answer_noise": answer_noise,
                                        "rep_seed": 0,
                                        "episode_id": ep,
                                        "mode_name": mode["mode_name"],
                                        "question_cost": mode["question_cost"],
                                        "ask_counts_as_step": mode["ask_counts_as_step"],
                                        "wrong_pick_fail": wrong_pick_fail,
                                        "deadline_margin": deadline_margin,
                                        "success": m["success"],
                                        "steps": m["steps"],
                                        "questions_asked": m["questions_asked"],
                                        "regret": m["regret"],
                                        "map_correct": m.get("final_map_correct", False),
                                        "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                                        "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                                        "failure_by_timeout": m.get("failure_by_timeout", False),
                                        "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                                        "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                                        "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                                        "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                                        "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                                    })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "eps",
                "answer_noise",
                "rep_seed",
                "episode_id",
                "mode_name",
                "question_cost",
                "ask_counts_as_step",
                "wrong_pick_fail",
                "deadline_margin",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows


def run_robust_answer_noise(output_csv="results/metrics_robust_answer_noise.csv"):
    """
    Robustness sweep A: vary answer noise at fixed principal/assistant dynamics.
    Focused on K={3,4} for runtime.
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    answer_noises = [0.0, 0.1, 0.2]
    principal_eps = config.DEFAULT_EPS
    assistant_eps = config.DEFAULT_EPS
    principal_beta = config.DEFAULT_BETA
    assistant_beta = config.DEFAULT_BETA
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

    rows = []
    for K in ks:
        for answer_noise in answer_noises:
            for policy in config.POLICIES:
                for rep_seed in reps:
                    for episode_id in range(n_episodes_per_seed):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 700000, rep_seed, K, principal_eps, principal_beta, episode_id
                        )
                        cfg = {
                            "ambiguity_K": K,
                            "principal_eps": principal_eps,
                            "assistant_eps": assistant_eps,
                            "principal_beta": principal_beta,
                            "assistant_beta": assistant_beta,
                            "answer_noise": answer_noise,
                        }
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy,
                            "K": K,
                            "answer_noise": answer_noise,
                            "principal_eps": principal_eps,
                            "assistant_eps": assistant_eps,
                            "principal_beta": principal_beta,
                            "assistant_beta": assistant_beta,
                            "rep_seed": int(rep_seed),
                            "episode_id": int(episode_id),
                            "success": m["success"],
                            "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"],
                            "map_correct": m.get("final_map_correct", False),
                            "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                            "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                            "failure_by_timeout": m.get("failure_by_timeout", False),
                            "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                            "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                            "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                            "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                            "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                        })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "answer_noise",
                "principal_eps",
                "assistant_eps",
                "principal_beta",
                "assistant_beta",
                "rep_seed",
                "episode_id",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows


def run_robust_mismatch(output_csv="results/metrics_robust_mismatch.csv"):
    """
    Robustness sweep B: model mismatch by varying principal beta while assistant beta is fixed.
    Focused on K={3,4} for runtime.
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    principal_betas = [1.0, 2.0, 4.0]
    principal_eps = config.DEFAULT_EPS
    assistant_eps = config.DEFAULT_EPS
    assistant_beta = config.DEFAULT_BETA
    answer_noise = config.ANSWER_NOISE
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = min(
        5, int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    )

    rows = []
    for K in ks:
        for principal_beta in principal_betas:
            for policy in config.POLICIES:
                for rep_seed in reps:
                    for episode_id in range(n_episodes_per_seed):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 800000, rep_seed, K, principal_eps, principal_beta, episode_id
                        )
                        cfg = {
                            "ambiguity_K": K,
                            "principal_eps": principal_eps,
                            "assistant_eps": assistant_eps,
                            "principal_beta": principal_beta,
                            "assistant_beta": assistant_beta,
                            "answer_noise": answer_noise,
                        }
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy,
                            "K": K,
                            "principal_beta": principal_beta,
                            "assistant_beta": assistant_beta,
                            "principal_eps": principal_eps,
                            "assistant_eps": assistant_eps,
                            "answer_noise": answer_noise,
                            "rep_seed": int(rep_seed),
                            "episode_id": int(episode_id),
                            "success": m["success"],
                            "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"],
                            "map_correct": m.get("final_map_correct", False),
                            "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                            "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                            "failure_by_timeout": m.get("failure_by_timeout", False),
                            "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                            "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                            "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                            "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                            "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                        })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "principal_beta",
                "assistant_beta",
                "principal_eps",
                "assistant_eps",
                "answer_noise",
                "rep_seed",
                "episode_id",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows


def run_question_difficulty_sweep(output_csv="results/metrics_question_difficulty.csv"):
    """
    Focused sweep for question-difficulty effects under mixed question types.
    Fixed K={3,4}, comparing:
      ask_or_act, info_gain_ask, easy_info_gain_ask, random_ask.
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    policies = ["ask_or_act", "info_gain_ask", "easy_info_gain_ask", "random_ask"]
    eps_vals = [config.DEFAULT_EPS]
    beta_vals = [config.DEFAULT_BETA]
    answer_noise = config.ANSWER_NOISE
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = min(
        2, int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    )

    total_per_condition = len(reps) * n_episodes_per_seed
    print(
        "[question_difficulty] episodes per (policy,K,eps,beta):",
        total_per_condition,
        f"({len(reps)} reps x {n_episodes_per_seed} eps/rep)",
    )

    rows = []
    for K in ks:
        for eps in eps_vals:
            for beta in beta_vals:
                for policy in policies:
                    for rep_seed in reps:
                        for episode_id in range(n_episodes_per_seed):
                            seed = _sweep_env_seed(
                                config.BASE_SEED + 900000, rep_seed, K, eps, beta, episode_id
                            )
                            cfg = {
                                "ambiguity_K": K,
                                "eps": eps,
                                "beta": beta,
                                "answer_noise": answer_noise,
                            }
                            m = _run_episode_impl(policy, seed, cfg)
                            rows.append({
                                "policy": policy,
                                "K": K,
                                "eps": eps,
                                "beta": beta,
                                "answer_noise": answer_noise,
                                "rep_seed": int(rep_seed),
                                "episode_id": int(episode_id),
                                "success": m["success"],
                                "steps": m["steps"],
                                "questions_asked": m["questions_asked"],
                                "regret": m["regret"],
                                "map_correct": m.get("final_map_correct", False),
                                "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                                "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                                "failure_by_timeout": m.get("failure_by_timeout", False),
                                "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                                "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                                "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                                "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                                "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                            })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "eps",
                "beta",
                "answer_noise",
                "rep_seed",
                "episode_id",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows


def _template_split_ids(holdout_ratio=0.30):
    n = len(TEMPLATES)
    all_ids = list(range(n))
    n_hold = max(1, int(round(float(holdout_ratio) * n)))
    holdout_ids = all_ids[-n_hold:]
    train_like_ids = all_ids[:-n_hold] if n_hold < n else []
    return train_like_ids, holdout_ids


def run_generalization_templates(output_csv="results/metrics_generalization_templates.csv"):
    """
    Held-out instruction-template evaluation.
    No training is performed; this probes brittleness under unseen templates.
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = list(config.AMBIGUITY_LEVELS)
    eps_vals = list(config.EPS_LEVELS)
    beta_vals = list(config.BETA_LEVELS)
    policies = ["ask_or_act", "info_gain_ask", "easy_info_gain_ask", "random_ask"]
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = int(
        getattr(
            config,
            "GENERALIZATION_EPISODES_PER_SEED",
            getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION),
        )
    )
    train_like_ids, holdout_ids = _template_split_ids(holdout_ratio=0.30)

    total_per_condition = len(reps) * n_episodes_per_seed
    print(
        "[generalization_templates] episodes per (policy,K,eps,beta):",
        total_per_condition,
        f"({len(reps)} reps x {n_episodes_per_seed} eps/rep), holdout_templates={holdout_ids}, train_like={train_like_ids}",
    )

    rows = []
    for K in ks:
        for eps in eps_vals:
            for beta in beta_vals:
                for policy in policies:
                    for rep_seed in reps:
                        for episode_id in range(n_episodes_per_seed):
                            seed = _sweep_env_seed(
                                config.BASE_SEED + 1000000, rep_seed, K, eps, beta, episode_id
                            )
                            cfg = {
                                "ambiguity_K": K,
                                "eps": eps,
                                "beta": beta,
                                "answer_noise": config.ANSWER_NOISE,
                                "allowed_template_ids": holdout_ids,
                            }
                            m = _run_episode_impl(policy, seed, cfg)
                            rows.append({
                                "policy": policy,
                                "K": K,
                                "eps": eps,
                                "beta": beta,
                                "answer_noise": config.ANSWER_NOISE,
                                "rep_seed": int(rep_seed),
                                "episode_id": int(episode_id),
                                "template_id": int(m.get("template_id", -1)),
                                "template_split": "heldout",
                                "heldout_template_ids": ",".join(str(x) for x in holdout_ids),
                                "success": m["success"],
                                "steps": m["steps"],
                                "questions_asked": m["questions_asked"],
                                "regret": m["regret"],
                                "map_correct": m.get("final_map_correct", False),
                                "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                                "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                                "failure_by_timeout": m.get("failure_by_timeout", False),
                                "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                                "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                                "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                                "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                                "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                            })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "eps",
                "beta",
                "answer_noise",
                "rep_seed",
                "episode_id",
                "template_id",
                "template_split",
                "heldout_template_ids",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows


def run_scale_k(output_csv="results/metrics_scaleK.csv"):
    """
    Scale ambiguity K up to 6 at fixed dynamics to keep runtime bounded.
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [1, 2, 3, 4, 5, 6]
    eps = 0.05
    beta = 2.0
    policies = list(config.POLICIES)
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = int(
        getattr(
            config,
            "SCALEK_EPISODES_PER_SEED",
            getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION),
        )
    )
    total_per_condition = len(reps) * n_episodes_per_seed
    print(
        "[scale_k] episodes per (policy,K):",
        total_per_condition,
        f"({len(reps)} reps x {n_episodes_per_seed} eps/rep), eps={eps}, beta={beta}",
    )

    rows = []
    for K in ks:
        for policy in policies:
            for rep_seed in reps:
                for episode_id in range(n_episodes_per_seed):
                    seed = _sweep_env_seed(
                        config.BASE_SEED + 1100000, rep_seed, K, eps, beta, episode_id
                    )
                    cfg = {
                        "ambiguity_K": K,
                        "eps": eps,
                        "beta": beta,
                        "answer_noise": config.ANSWER_NOISE,
                    }
                    if policy == "pomcp_planner":
                        cfg["pomcp_iters"] = 40
                        cfg["pomcp_horizon"] = 6
                    m = _run_episode_impl(policy, seed, cfg)
                    rows.append({
                        "policy": policy,
                        "K": K,
                        "eps": eps,
                        "beta": beta,
                        "answer_noise": config.ANSWER_NOISE,
                        "rep_seed": int(rep_seed),
                        "episode_id": int(episode_id),
                        "success": m["success"],
                        "steps": m["steps"],
                        "questions_asked": m["questions_asked"],
                        "regret": m["regret"],
                        "map_correct": m.get("final_map_correct", False),
                        "episode_max_steps": m.get("episode_max_steps", config.MAX_STEPS),
                        "failure_by_wrong_pick": m.get("failure_by_wrong_pick", False),
                        "failure_by_timeout": m.get("failure_by_timeout", False),
                        "entropy_before_first_ask": m.get("entropy_before_first_ask", 0.0),
                        "entropy_after_first_ask": m.get("entropy_after_first_ask", 0.0),
                        "effective_goal_count_before": m.get("effective_goal_count_before", 0.0),
                        "effective_goal_count_after": m.get("effective_goal_count_after", 0.0),
                        "ig_of_first_asked_question": m.get("ig_of_first_asked_question", 0.0),
                    })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "K",
                "eps",
                "beta",
                "answer_noise",
                "rep_seed",
                "episode_id",
                "success",
                "steps",
                "questions_asked",
                "regret",
                "map_correct",
                "episode_max_steps",
                "failure_by_wrong_pick",
                "failure_by_timeout",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows
