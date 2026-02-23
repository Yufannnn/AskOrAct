"""Evaluation: run_episode, run_sweep, run_ablations. Optional parallel workers."""

import csv
import json
import os
import numpy as np
from src import config
from src.env import grid_distance
from src.world import generate_world, instruction_to_candidate_goals, answer_question, answer_likelihood, list_questions
from src.inference import init_posterior, update_posterior, posterior_entropy
from src.agents import sample_principal_action, policy_ask_or_act, policy_never_ask, policy_always_ask, _bfs_next_step
from src.agents.assistant import CostAct, best_question_cost, assistant_task_action


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
    answer_noise = config_dict.get("answer_noise", config.ANSWER_NOISE)
    N = config_dict.get("N", config.DEFAULT_N)
    M = config_dict.get("M", config.DEFAULT_M)

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

    env, instruction_u, true_goal_obj_id = generate_world(seed, N=N, M=M, ambiguity_K=K)
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
        }

    posterior = init_posterior(candidate_goals)
    principal_action_history = []
    questions_asked = 0
    asked_qnames = set()
    steps = 0
    success = False
    terminated_by_wrong_pick = False

    policy_fn = {
        "ask_or_act": policy_ask_or_act,
        "never_ask": policy_never_ask,
        "always_ask": policy_always_ask,
    }[policy_name]

    episode_q_cap = _episode_question_cap(max_questions_per_episode)
    policy_max_questions = config.MAX_QUESTIONS
    if episode_q_cap is not None:
        policy_max_questions = min(policy_max_questions, episode_q_cap)

    while steps < episode_max_steps:
        state = env.get_state()
        principal_action = sample_principal_action(state, true_goal_obj_id, env, rng, beta, eps)
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
            "beta": beta,
            "eps": eps,
            "answer_noise": answer_noise,
            "question_cost": question_cost,
            "max_questions": policy_max_questions,
            "entropy_threshold": config.ENTROPY_THRESHOLD,
            "entropy_gate": config.ENTROPY_GATE,
            "ask_window": config.ASK_WINDOW,
            "asked_qnames": asked_qnames,
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
                continue

        assistant_action = out
        update_posterior(posterior, state, principal_action, candidate_goals, env, beta, eps)
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
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("Wrote", output_csv, f"({len(rows)} rows)")
    return rows
