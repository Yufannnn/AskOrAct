"""Evaluation: run_episode, run_sweep. Optional parallel workers."""

import csv
import json
import os
import numpy as np
from src import config
from src.world import generate_world, instruction_to_candidate_goals, answer_question, answer_likelihood, list_questions
from src.inference import init_posterior, update_posterior
from src.agents import sample_principal_action, policy_ask_or_act, policy_never_ask, policy_always_ask, _bfs_next_step


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
            _, done, info = env.step("stay", "pick", true_goal_obj_id=true_goal_obj_id)
            steps += 1
            return steps
        act = _bfs_next_step(env.N, env.walls, a_pos, goal_pos)
        env.step("stay", act, true_goal_obj_id=true_goal_obj_id)
        steps += 1
        state = env.get_state()
    return steps


def _run_episode_impl(policy_name, seed, config_dict):
    """Single episode; used by run_episode and parallel workers."""
    K = config_dict.get("ambiguity_K", 2)
    eps = config_dict.get("eps", config.DEFAULT_EPS)
    beta = config_dict.get("beta", config.DEFAULT_BETA)
    answer_noise = config_dict.get("answer_noise", config.ANSWER_NOISE)
    N = config_dict.get("N", config.DEFAULT_N)
    M = config_dict.get("M", config.DEFAULT_M)
    rng = np.random.default_rng(seed)

    env, instruction_u, true_goal_obj_id = generate_world(seed, N=N, M=M, ambiguity_K=K)
    candidate_goals = instruction_to_candidate_goals(instruction_u, env)
    if not candidate_goals:
        return {"success": False, "steps": config.MAX_STEPS, "questions_asked": 0, "assistant_picked_goal": False, "final_map_correct": False, "regret": config.MAX_STEPS, "oracle_steps": 0}

    posterior = init_posterior(candidate_goals)
    principal_action_history = []
    questions_asked = 0
    steps = 0
    success = False

    policy_fn = {"ask_or_act": policy_ask_or_act, "never_ask": policy_never_ask, "always_ask": policy_always_ask}[policy_name]

    while steps < config.MAX_STEPS:
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
            "max_questions": config.MAX_QUESTIONS,
            "entropy_threshold": config.ENTROPY_THRESHOLD,
        }
        out, posterior = policy_fn(**policy_kw)

        if isinstance(out, tuple) and out[0] == "ask":
            _, q_id = out
            questions_asked += 1
            steps += 1
            q_tuple = next((q for q in list_questions() if q[0] == q_id), None)
            if q_tuple is None:
                continue
            ans = answer_question(q_tuple, true_goal_obj_id, env, rng, answer_noise)
            p_new = {g: posterior.get(g, 0) * answer_likelihood(q_tuple, ans, g, env, answer_noise) for g in candidate_goals}
            Z = sum(p_new.values())
            if Z > 0:
                for g in candidate_goals:
                    posterior[g] = p_new[g] / Z
            continue

        assistant_action = out
        update_posterior(posterior, state, principal_action, candidate_goals, env, beta, eps)
        state, done, info = env.step(principal_action, assistant_action, true_goal_obj_id=true_goal_obj_id)
        steps += 1
        success = info.get("assistant_picked_goal", False)
        if done or success:
            break

    final_map_correct = (max(candidate_goals, key=lambda g: posterior.get(g, 0)) == true_goal_obj_id) if candidate_goals else False
    env_oracle, _, _ = generate_world(seed, N=N, M=M, ambiguity_K=K)
    o_steps = oracle_steps(env_oracle, true_goal_obj_id, np.random.default_rng(seed + 1))
    regret = (steps - o_steps) if success else (config.MAX_STEPS - o_steps)

    return {
        "success": success,
        "steps": steps,
        "questions_asked": questions_asked,
        "assistant_picked_goal": success,
        "final_map_correct": final_map_correct,
        "regret": regret,
        "oracle_steps": o_steps,
    }


def run_episode(policy_name, seed, config_dict):
    """Run one episode. config_dict can override config (ambiguity_K, eps, beta, etc.)."""
    return _run_episode_impl(policy_name, seed, config_dict)


def run_sweep(output_csv="results/metrics.csv", output_json="results/summary.json"):
    """Sweep over conditions; optionally use parallel workers. Writes CSV and JSON."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    rows = []
    n_workers = getattr(config, "N_WORKERS", 0)

    if n_workers and n_workers > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = []
        for K in config.AMBIGUITY_LEVELS:
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in config.POLICIES:
                        for ep in range(config.N_EPISODES_PER_CONDITION):
                            seed = config.BASE_SEED + K * 1000 + int(eps * 100) + int(beta * 10) + (config.POLICIES.index(policy_name) * 1000) + ep
                            tasks.append((policy_name, seed, cfg))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_episode_impl, p, s, c): (p, s, c) for p, s, c in tasks}
            for fut in as_completed(futures):
                policy_name, seed, cfg = futures[fut]
                try:
                    m = fut.result()
                except Exception as e:
                    m = {"success": False, "steps": config.MAX_STEPS, "questions_asked": 0, "assistant_picked_goal": False, "final_map_correct": False, "regret": config.MAX_STEPS, "oracle_steps": 0}
                K, eps, beta = cfg["ambiguity_K"], cfg["eps"], cfg["beta"]
                ep = seed - config.BASE_SEED - K * 1000 - int(eps * 100) - int(beta * 10) - (config.POLICIES.index(policy_name) * 1000)
                ep = ep % config.N_EPISODES_PER_CONDITION
                rows.append({
                    "ambiguity_K": K, "eps": eps, "beta": beta, "policy": policy_name, "episode": ep,
                    "success": m["success"], "steps": m["steps"], "questions_asked": m["questions_asked"],
                    "assistant_picked_goal": m.get("assistant_picked_goal", m["success"]), "final_map_correct": m.get("final_map_correct", False),
                    "oracle_steps": m.get("oracle_steps", 0), "regret": m["regret"],
                })
    else:
        for K in config.AMBIGUITY_LEVELS:
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in config.POLICIES:
                        for ep in range(config.N_EPISODES_PER_CONDITION):
                            seed = config.BASE_SEED + K * 1000 + int(eps * 100) + int(beta * 10) + (config.POLICIES.index(policy_name) * 1000) + ep
                            m = _run_episode_impl(policy_name, seed, cfg)
                            rows.append({
                                "ambiguity_K": K, "eps": eps, "beta": beta, "policy": policy_name, "episode": ep,
                                "success": m["success"], "steps": m["steps"], "questions_asked": m["questions_asked"],
                                "assistant_picked_goal": m.get("assistant_picked_goal", m["success"]), "final_map_correct": m.get("final_map_correct", False),
                                "oracle_steps": m.get("oracle_steps", 0), "regret": m["regret"],
                            })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ambiguity_K", "eps", "beta", "policy", "episode", "success", "steps", "questions_asked", "assistant_picked_goal", "final_map_correct", "oracle_steps", "regret"])
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for K in config.AMBIGUITY_LEVELS:
        for eps in config.EPS_LEVELS:
            for beta in config.BETA_LEVELS:
                key = (K, eps, beta)
                summary[key] = {}
                for policy_name in config.POLICIES:
                    sub = [r for r in rows if r["ambiguity_K"] == K and r["eps"] == eps and r["beta"] == beta and r["policy"] == policy_name]
                    summary[key][policy_name] = {
                        "success_rate": np.mean([r["success"] for r in sub]),
                        "avg_steps": np.mean([r["steps"] for r in sub]),
                        "avg_questions": np.mean([r["questions_asked"] for r in sub]),
                        "avg_regret": np.mean([r["regret"] for r in sub]),
                    }
    with open(output_json, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    return rows, summary
