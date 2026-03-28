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
from src.agents.assistant import (
    CostAct, best_question_cost, assistant_task_action, question_info_gain,
    policy_passive_aware_ask, policy_cost_wait_ask, policy_mismatch_aware_ask,
)


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


def _policies_for_main_sweep(ambiguity_K):
    policies = list(getattr(config, "MAIN_SWEEP_BASE_POLICIES", config.POLICIES))
    extra_by_k = getattr(config, "MAIN_SWEEP_EXTRA_POLICIES_BY_K", {})
    for policy_name in extra_by_k.get(int(ambiguity_K), []):
        if policy_name not in policies:
            policies.append(policy_name)
    return policies


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

    layout_type = config_dict.get("layout_type", getattr(config, "DEFAULT_LAYOUT_TYPE", "vertical"))
    prior_type = config_dict.get("prior_type", getattr(config, "DEFAULT_PRIOR_TYPE", "uniform"))
    two_rooms = config_dict.get("two_rooms", None)
    wrong_pick_penalty = config_dict.get("wrong_pick_penalty", getattr(config, "WRONG_PICK_PENALTY", 0))
    mismatch_true_noise_by_qtype = config_dict.get("mismatch_true_noise_by_qtype", None)
    action_drop_rate = config_dict.get("action_drop_rate", 0.0)
    temper_passive = config_dict.get("temper_passive", False)
    temper_gamma = config_dict.get("temper_gamma", 0.90)
    temper_baseline = config_dict.get("temper_baseline", 1.5)
    temper_scale = config_dict.get("temper_scale", 3.0)  # omega = 1/(1 + S_t/scale)

    env, instruction_u, true_goal_obj_id = generate_world(
        seed, N=N, M=M, ambiguity_K=K, allowed_template_ids=allowed_template_ids,
        layout_type=layout_type, two_rooms=two_rooms,
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
            "entropy_before_first_ask": float("nan"),
            "entropy_after_first_ask": float("nan"),
            "effective_goal_count_before": float("nan"),
            "effective_goal_count_after": float("nan"),
            "ig_of_first_asked_question": float("nan"),
            "template_id": -1,
        }

    posterior = init_posterior(
        candidate_goals, prior_type=prior_type, env=env,
        assistant_pos=env.get_state()["assistant_pos"],
    )
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
        "passive_aware_ask": policy_passive_aware_ask,
        "cost_wait_ask": policy_cost_wait_ask,
        "mismatch_aware_ask": policy_mismatch_aware_ask,
    }[policy_name]

    _surprise_state = {"S": 0.0}  # persistent surprise state for MismatchAwareAsk

    episode_q_cap = _episode_question_cap(max_questions_per_episode)
    policy_max_questions = config.MAX_QUESTIONS
    if episode_q_cap is not None:
        policy_max_questions = min(policy_max_questions, episode_q_cap)

    while steps < episode_max_steps:
        state = env.get_state()
        principal_action = sample_principal_action(
            state, true_goal_obj_id, env, rng, principal_beta, principal_eps
        )
        # NOTE: principal_action_history is appended only after a real env step,
        # not on ask turns, so the ask-window counter reflects actual progression.

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
            "ask_window": config_dict.get("ask_window_override", config.ASK_WINDOW),
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
            "wrong_pick_penalty": wrong_pick_penalty,
            "_surprise_state": _surprise_state,
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
                    print(f"[WARN] question id '{q_id}' not found in list_questions(); skipping")
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
                if mismatch_true_noise_by_qtype is not None:
                    _saved_noise = dict(config.ANSWER_NOISE_BY_QTYPE)
                    config.ANSWER_NOISE_BY_QTYPE = dict(mismatch_true_noise_by_qtype)
                ans = answer_question(q_tuple, true_goal_obj_id, env, rng, answer_noise)
                if mismatch_true_noise_by_qtype is not None:
                    config.ANSWER_NOISE_BY_QTYPE = _saved_noise
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
        if action_drop_rate > 0 and rng.random() < action_drop_rate:
            pass  # assistant misses this principal action — no posterior update
        else:
            principal_action_history.append(principal_action)
            # Compute tempering weight if enabled
            omega = 1.0
            if temper_passive and principal_action_history:
                import math
                from src.agents.principal import principal_action_probs as _pap
                # Predictive probability of observed action
                p_pred = sum(
                    max(0, posterior.get(g, 0)) * _pap(state, g, env, assistant_beta, assistant_eps).get(principal_action, 1e-10)
                    for g in candidate_goals
                )
                p_pred = max(p_pred, 1e-10)
                surprise = -math.log(p_pred)
                # CUSUM accumulator (stored in config_dict for persistence)
                S_prev = config_dict.get("_temper_S", 0.0)
                S_t = max(0.0, temper_gamma * S_prev + (surprise - temper_baseline))
                config_dict["_temper_S"] = S_t
                omega = 1.0 / (1.0 + S_t / temper_scale)
            update_posterior(
                posterior, state, principal_action, candidate_goals, env, assistant_beta, assistant_eps,
                omega=omega,
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
    """Sweep over the paper's main evaluation grid and write CSV/JSON summaries."""
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
            policy_names = _policies_for_main_sweep(K)
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in policy_names:
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
                    import traceback
                    traceback.print_exc()
                    print(f"[sweep] ERROR in episode policy={policy_name} seed={seed} cfg={cfg} — substituting failure row", flush=True)
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
            policy_names = _policies_for_main_sweep(K)
            for eps in config.EPS_LEVELS:
                for beta in config.BETA_LEVELS:
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    for policy_name in policy_names:
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
        policy_names = _policies_for_main_sweep(K)
        for eps in config.EPS_LEVELS:
            for beta in config.BETA_LEVELS:
                key = (K, eps, beta)
                summary[key] = {}
                for policy_name in policy_names:
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
                            for policy in config.ABLATION_POLICIES:
                                for ep in range(n_eps):
                                    seed = (
                                        900000
                                        + K * 10000
                                        + int(eps * 100) * 100
                                        + (0 if mode["mode_name"] == "modeA" else 1 if mode["mode_name"] == "modeB" else 2) * 1000
                                        + (100 if wrong_pick_fail else 0)
                                        + deadline_margin * 10
                                        + config.ABLATION_POLICIES.index(policy) * 100
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
    # Disable per-type config overrides so the swept answer_noise values are
    # actually used by _noise_for_question() instead of being shadowed by
    # ANSWER_NOISE_BY_QTYPE (which covers all question types).
    _saved_by_qtype = getattr(config, "ANSWER_NOISE_BY_QTYPE", {})
    ks = [3, 4]
    answer_noises = [0.0, 0.1, 0.2]
    principal_eps = config.DEFAULT_EPS
    assistant_eps = config.DEFAULT_EPS
    principal_beta = config.DEFAULT_BETA
    assistant_beta = config.DEFAULT_BETA
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

    try:
        config.ANSWER_NOISE_BY_QTYPE = {}
        rows = []
        for K in ks:
            for answer_noise in answer_noises:
                for policy in config.ROBUST_ANSWER_POLICIES:
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
    finally:
        config.ANSWER_NOISE_BY_QTYPE = _saved_by_qtype
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
    n_episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

    rows = []
    for K in ks:
        for principal_beta in principal_betas:
            for policy in config.ROBUST_MISMATCH_POLICIES:
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
    n_episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

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
    policies = list(config.SCALEK_POLICIES)
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


# ---------------------------------------------------------------------------
# Generalization experiment: layout type (vertical vs horizontal wall)
# ---------------------------------------------------------------------------

def run_generalization_layout(output_csv="results/metrics_generalization_layout.csv"):
    """Sweep layout type (vertical/horizontal wall) at K={3,4}."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    layout_types = ["vertical", "horizontal"]
    policies = getattr(config, "GENERALIZATION_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

    rows = []
    for K in ks:
        for layout in layout_types:
            for policy in policies:
                for eps in config.EPS_LEVELS:
                    for beta in config.BETA_LEVELS:
                        for rep_seed in reps:
                            for eid in range(n_ep):
                                seed = _sweep_env_seed(config.BASE_SEED + 900000, rep_seed, K, eps, beta, eid)
                                cfg = {"ambiguity_K": K, "eps": eps, "beta": beta, "layout_type": layout}
                                m = _run_episode_impl(policy, seed, cfg)
                                rows.append({
                                    "policy": policy, "K": K, "layout_type": layout,
                                    "eps": eps, "beta": beta,
                                    "rep_seed": int(rep_seed), "episode_id": int(eid),
                                    "success": m["success"], "steps": m["steps"],
                                    "questions_asked": m["questions_asked"],
                                    "regret": m["regret"],
                                    "team_cost": m["team_cost"],
                                })
    fieldnames = ["policy", "K", "layout_type", "eps", "beta", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Layout sweep:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Generalization experiment: non-uniform (distance-weighted) prior
# ---------------------------------------------------------------------------

def run_generalization_prior(output_csv="results/metrics_generalization_prior.csv"):
    """Sweep prior type (uniform/distance) at K={3,4}."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    prior_types = ["uniform", "distance"]
    policies = getattr(config, "GENERALIZATION_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))

    rows = []
    for K in ks:
        for prior in prior_types:
            for policy in policies:
                for eps in config.EPS_LEVELS:
                    for beta in config.BETA_LEVELS:
                        for rep_seed in reps:
                            for eid in range(n_ep):
                                seed = _sweep_env_seed(config.BASE_SEED + 950000, rep_seed, K, eps, beta, eid)
                                cfg = {"ambiguity_K": K, "eps": eps, "beta": beta, "prior_type": prior}
                                m = _run_episode_impl(policy, seed, cfg)
                                rows.append({
                                    "policy": policy, "K": K, "prior_type": prior,
                                    "eps": eps, "beta": beta,
                                    "rep_seed": int(rep_seed), "episode_id": int(eid),
                                    "success": m["success"], "steps": m["steps"],
                                    "questions_asked": m["questions_asked"],
                                    "regret": m["regret"],
                                    "team_cost": m["team_cost"],
                                })
    fieldnames = ["policy", "K", "prior_type", "eps", "beta", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Prior sweep:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Generalization experiment: asymmetric answer noise
# ---------------------------------------------------------------------------

def run_generalization_asymmetric_noise(output_csv="results/metrics_generalization_asymmetric_noise.csv"):
    """Sweep answer-noise profile (default symmetric / asymmetric) at K={3,4}."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    policies = getattr(config, "GENERALIZATION_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    noise_profiles = {
        "default": dict(config.ANSWER_NOISE_BY_QTYPE),
        "asymmetric": dict(getattr(config, "ASYMMETRIC_NOISE_BY_QTYPE", config.ANSWER_NOISE_BY_QTYPE)),
    }
    _saved_by_qtype = dict(config.ANSWER_NOISE_BY_QTYPE)

    rows = []
    try:
        for K in ks:
            for profile_name, profile_noise in noise_profiles.items():
                config.ANSWER_NOISE_BY_QTYPE = dict(profile_noise)
                for policy in policies:
                    for eps in config.EPS_LEVELS:
                        for beta in config.BETA_LEVELS:
                            for rep_seed in reps:
                                for eid in range(n_ep):
                                    seed = _sweep_env_seed(config.BASE_SEED + 980000, rep_seed, K, eps, beta, eid)
                                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                                    m = _run_episode_impl(policy, seed, cfg)
                                    rows.append({
                                        "policy": policy, "K": K, "noise_profile": profile_name,
                                        "eps": eps, "beta": beta,
                                        "rep_seed": int(rep_seed), "episode_id": int(eid),
                                        "success": m["success"], "steps": m["steps"],
                                        "questions_asked": m["questions_asked"],
                                        "regret": m["regret"],
                                        "team_cost": m["team_cost"],
                                    })
    finally:
        config.ANSWER_NOISE_BY_QTYPE = _saved_by_qtype

    fieldnames = ["policy", "K", "noise_profile", "eps", "beta", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Noise sweep:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Structural OOD: different grid size + single room
# ---------------------------------------------------------------------------

def run_structural_ood(output_csv="results/metrics_structural_ood.csv"):
    """Sweep grid size (7,9,11) x room config (two-room, single) at K={3,4}."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    grid_sizes = getattr(config, "STRUCTURAL_OOD_GRID_SIZES", [7, 9, 11])
    room_configs = getattr(config, "STRUCTURAL_OOD_ROOM_CONFIGS", [True, False])
    policies = getattr(config, "STRUCTURAL_OOD_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    rows = []
    for K in ks:
        for N in grid_sizes:
            for two_rooms in room_configs:
                for policy in policies:
                    for rep_seed in reps:
                        for eid in range(n_ep):
                            seed = _sweep_env_seed(
                                config.BASE_SEED + 1200000, rep_seed, K, eps, beta, eid
                            ) + N * 7 + (0 if two_rooms else 3)
                            cfg = {
                                "ambiguity_K": K, "eps": eps, "beta": beta,
                                "N": N, "two_rooms": two_rooms,
                            }
                            m = _run_episode_impl(policy, seed, cfg)
                            rows.append({
                                "policy": policy, "K": K, "N": N,
                                "two_rooms": two_rooms,
                                "rep_seed": int(rep_seed), "episode_id": int(eid),
                                "success": m["success"], "steps": m["steps"],
                                "questions_asked": m["questions_asked"],
                                "regret": m["regret"],
                                "team_cost": m["team_cost"],
                            })
    fieldnames = ["policy", "K", "N", "two_rooms", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Structural OOD:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Model-mismatch (extended): noise mismatch + prior mismatch
# ---------------------------------------------------------------------------

def run_model_mismatch_extended(output_csv="results/metrics_model_mismatch_ext.csv"):
    """
    Extended model-mismatch: noise mismatch (agent assumes flat noise while true
    noise is per-qtype) and prior mismatch (agent uses distance prior on uniform world).
    """
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    policies = getattr(config, "MISMATCH_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    # True per-qtype noise profile
    true_noise_profile = dict(config.ANSWER_NOISE_BY_QTYPE)

    # Conditions: (label, true_noise_by_qtype, agent_noise_by_qtype, agent_answer_noise, agent_prior)
    conditions = [
        # Matched baseline
        ("matched", None, None, config.ANSWER_NOISE, "uniform"),
        # Noise mismatch: true per-qtype, agent assumes flat 0.10
        ("noise_flat010", true_noise_profile, {}, 0.10, "uniform"),
        # Noise mismatch: true per-qtype, agent assumes zero noise
        ("noise_flat000", true_noise_profile, {}, 0.0, "uniform"),
        # Prior mismatch: agent uses distance prior on uniformly-placed goals
        ("prior_distance", None, None, config.ANSWER_NOISE, "distance"),
    ]

    _saved_by_qtype = dict(config.ANSWER_NOISE_BY_QTYPE)
    rows = []
    try:
        for K in ks:
            for cond_label, true_nq, agent_nq, agent_an, agent_prior in conditions:
                # Set agent's believed noise profile
                if agent_nq is not None:
                    config.ANSWER_NOISE_BY_QTYPE = dict(agent_nq)
                else:
                    config.ANSWER_NOISE_BY_QTYPE = dict(_saved_by_qtype)
                for policy in policies:
                    for rep_seed in reps:
                        for eid in range(n_ep):
                            seed = _sweep_env_seed(
                                config.BASE_SEED + 1300000, rep_seed, K, eps, beta, eid
                            )
                            cfg = {
                                "ambiguity_K": K, "eps": eps, "beta": beta,
                                "answer_noise": agent_an,
                                "prior_type": agent_prior,
                            }
                            if true_nq is not None:
                                cfg["mismatch_true_noise_by_qtype"] = true_nq
                            m = _run_episode_impl(policy, seed, cfg)
                            rows.append({
                                "policy": policy, "K": K,
                                "condition": cond_label,
                                "rep_seed": int(rep_seed), "episode_id": int(eid),
                                "success": m["success"], "steps": m["steps"],
                                "questions_asked": m["questions_asked"],
                                "regret": m["regret"],
                                "team_cost": m["team_cost"],
                            })
    finally:
        config.ANSWER_NOISE_BY_QTYPE = _saved_by_qtype

    fieldnames = ["policy", "K", "condition", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Model-mismatch extended:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Failure penalty sweep: wrong-pick penalty shifts the ask threshold
# ---------------------------------------------------------------------------

def run_failure_penalty_sweep(output_csv="results/metrics_failure_penalty.csv"):
    """Sweep wrong-pick penalty c_fail at K={3,4} to show ask-threshold shift."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    penalties = getattr(config, "FAILURE_PENALTY_VALUES", [0, 5, 10, 20, 50])
    policies = getattr(config, "FAILURE_PENALTY_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    rows = []
    for K in ks:
        for c_fail in penalties:
            for policy in policies:
                for rep_seed in reps:
                    for eid in range(n_ep):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 1400000, rep_seed, K, eps, beta, eid
                        )
                        cfg = {
                            "ambiguity_K": K, "eps": eps, "beta": beta,
                            "wrong_pick_penalty": c_fail,
                        }
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy, "K": K,
                            "c_fail": c_fail,
                            "rep_seed": int(rep_seed), "episode_id": int(eid),
                            "success": m["success"], "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"],
                            "team_cost": m["team_cost"],
                        })
    fieldnames = ["policy", "K", "c_fail", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Failure penalty:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Action-drop sweep: degrade passive channel quality
# ---------------------------------------------------------------------------

def run_action_drop_sweep(output_csv="results/metrics_action_drop.csv"):
    """Sweep action drop rate to measure passive channel degradation effect."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    drop_rates = getattr(config, "ACTION_DROP_RATES", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    policies = getattr(config, "ACTION_DROP_POLICIES", ["ask_or_act", "never_ask", "info_gain_ask"])
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    rows = []
    for K in ks:
        for p_drop in drop_rates:
            for policy in policies:
                for rep_seed in reps:
                    for eid in range(n_ep):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 1600000, rep_seed, K, eps, beta, eid
                        )
                        cfg = {
                            "ambiguity_K": K, "eps": eps, "beta": beta,
                            "action_drop_rate": p_drop,
                        }
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy, "K": K,
                            "action_drop_rate": p_drop,
                            "rep_seed": int(rep_seed), "episode_id": int(eid),
                            "success": m["success"], "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"],
                            "team_cost": m["team_cost"],
                        })
    fieldnames = ["policy", "K", "action_drop_rate", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Action drop:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Cost × Passive-quality heatmap
# ---------------------------------------------------------------------------

def run_cost_passive_heatmap(output_csv="results/metrics_cost_passive_heatmap.csv"):
    """2D sweep: question cost × action drop rate at K=4, ask_or_act only."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    K = 4
    cq_levels = getattr(config, "HEATMAP_CQ_LEVELS", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5])
    drop_levels = getattr(config, "HEATMAP_DROP_LEVELS", [0.0, 0.2, 0.4, 0.6, 0.8])
    policy = "ask_or_act"
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    rows = []
    for cq in cq_levels:
        for p_drop in drop_levels:
            for rep_seed in reps:
                for eid in range(n_ep):
                    seed = _sweep_env_seed(
                        config.BASE_SEED + 1700000, rep_seed, K, eps, beta, eid
                    )
                    cfg = {
                        "ambiguity_K": K, "eps": eps, "beta": beta,
                        "question_cost": cq,
                        "action_drop_rate": p_drop,
                    }
                    m = _run_episode_impl(policy, seed, cfg)
                    rows.append({
                        "policy": policy, "K": K,
                        "question_cost": cq,
                        "action_drop_rate": p_drop,
                        "rep_seed": int(rep_seed), "episode_id": int(eid),
                        "success": m["success"], "steps": m["steps"],
                        "questions_asked": m["questions_asked"],
                        "regret": m["regret"],
                        "team_cost": m["team_cost"],
                    })
    fieldnames = ["policy", "K", "question_cost", "action_drop_rate", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Cost×Passive heatmap:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Novelty Experiment A: Passive-resolvability analysis
# Measure shared optimal first action among candidate goals per episode
# ---------------------------------------------------------------------------

def run_passive_resolvability(output_csv="results/metrics_passive_resolvability.csv"):
    """Per-episode: compute path overlap among candidate goals, then correlate with ask behavior."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    from src.agents.assistant import _bfs_next_step
    ks = [3, 4]
    policies = ["ask_or_act", "never_ask", "info_gain_ask"]
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS
    beta = config.DEFAULT_BETA

    rows = []
    for K in ks:
        for policy in policies:
            for rep_seed in reps:
                for eid in range(n_ep):
                    seed = _sweep_env_seed(config.BASE_SEED + 1800000, rep_seed, K, eps, beta, eid)
                    cfg = {"ambiguity_K": K, "eps": eps, "beta": beta}
                    # Generate world to measure overlap
                    env_tmp, instr_tmp, true_g = generate_world(seed, ambiguity_K=K)
                    cands = instruction_to_candidate_goals(instr_tmp, env_tmp)
                    state_tmp = env_tmp.get_state()
                    p_pos = state_tmp["principal_pos"]
                    # Compute first optimal action for each candidate from principal position
                    first_actions = []
                    for g in cands:
                        obj = env_tmp.get_object_by_id(g)
                        if obj and not obj.get("collected", False):
                            a = _bfs_next_step(env_tmp.N, env_tmp.walls, p_pos, obj["pos"])
                            first_actions.append(a)
                    # Overlap: fraction of candidates sharing the modal first action
                    if first_actions:
                        from collections import Counter
                        ctr = Counter(first_actions)
                        modal_count = ctr.most_common(1)[0][1]
                        overlap = modal_count / len(first_actions)
                    else:
                        overlap = 1.0
                    # Now run the episode
                    m = _run_episode_impl(policy, seed, cfg)
                    rows.append({
                        "policy": policy, "K": K, "overlap": round(overlap, 3),
                        "rep_seed": int(rep_seed), "episode_id": int(eid),
                        "success": m["success"], "steps": m["steps"],
                        "questions_asked": m["questions_asked"],
                        "regret": m["regret"], "team_cost": m["team_cost"],
                    })
    fieldnames = ["policy", "K", "overlap", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Passive resolvability:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Novelty Experiment B: Informative irrationality (beta inverted-U)
# ---------------------------------------------------------------------------

def run_informative_irrationality(output_csv="results/metrics_informative_irrationality.csv"):
    """Fine-grained beta sweep to find inverted-U in assistance quality."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    policies = ["ask_or_act", "never_ask", "info_gain_ask"]
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS

    rows = []
    for K in ks:
        for true_beta in betas:
            for policy in policies:
                for rep_seed in reps:
                    for eid in range(n_ep):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 1900000, rep_seed, K, eps, true_beta, eid
                        )
                        cfg = {
                            "ambiguity_K": K, "eps": eps,
                            "principal_beta": true_beta,
                            "assistant_beta": true_beta,  # matched
                            "beta": true_beta,
                        }
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy, "K": K, "beta": true_beta,
                            "rep_seed": int(rep_seed), "episode_id": int(eid),
                            "success": m["success"], "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"], "team_cost": m["team_cost"],
                        })
    fieldnames = ["policy", "K", "beta", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Informative irrationality:", output_csv, f"({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# Novelty Experiment C: More observation can hurt under mismatch
# Force h observation steps before allowing ask/act, with mismatched beta
# ---------------------------------------------------------------------------

def run_observation_hurts_mismatch(output_csv="results/metrics_obs_hurts_mismatch.csv"):
    """Force observation horizon h, then allow normal policy. Matched vs mismatched beta."""
    os.makedirs(os.path.dirname(output_csv) or "results", exist_ok=True)
    ks = [3, 4]
    obs_horizons = [0, 1, 2, 3, 4, 5]
    # Matched: assistant_beta = principal_beta = 2
    # Mismatched: principal_beta = 1 (near-random), assistant_beta = 2
    mismatch_conditions = [
        ("matched", 2.0, 2.0),
        ("mismatch_b1", 1.0, 2.0),
        ("mismatch_b4", 4.0, 2.0),
    ]
    policy = "ask_or_act"
    reps = list(getattr(config, "REPL_SEEDS", [0]))
    n_ep = int(getattr(config, "N_EPISODES_PER_SEED", config.N_EPISODES_PER_CONDITION))
    eps = config.DEFAULT_EPS

    rows = []
    for K in ks:
        for h in obs_horizons:
            for cond_name, p_beta, a_beta in mismatch_conditions:
                for rep_seed in reps:
                    for eid in range(n_ep):
                        seed = _sweep_env_seed(
                            config.BASE_SEED + 2000000, rep_seed, K, eps, p_beta, eid
                        )
                        # Force observation: set ask_window to start after h steps
                        # ask_window = 6 means ask allowed in steps 0-6
                        # To force h observe steps: set ask_window = max(0, 6 - h)
                        # But simpler: just don't allow asking in first h steps
                        # We use the ask_window parameter creatively:
                        # Normal ask_window = 6. We set it so asking starts at step h.
                        # Actually, ask_window blocks asking AFTER step ask_window.
                        # To block asking BEFORE step h, we need a different mechanism.
                        # Simplest: for h>0, run first h steps as never_ask, then switch.
                        # But that requires modifying the episode loop.
                        # Alternative: use action_drop_rate=0 (observe everything) but
                        # set max_questions_per_episode based on h.
                        # Actually, simplest correct approach: set ask_window = 6 - h
                        # This means the policy can only ask during steps 0..(6-h).
                        # With h=0: window=6 (normal). h=3: window=3. h=6: window=0 (never ask).
                        # This isn't exactly "force observe h then ask" but it reduces
                        # the ask window, which has a similar effect.
                        # Better: just use the existing machinery. The ask_window check is:
                        #   step_t = len(principal_action_history)
                        #   if step_t > ask_window: don't ask
                        # So if I set ask_window = 6 but also set a min_ask_step,
                        # I'd need to modify the policy.
                        # For now, let's just use ask_window = max(0, 6 - h) as a proxy.
                        effective_window = max(0, 6 - h)
                        cfg = {
                            "ambiguity_K": K, "eps": eps,
                            "principal_beta": p_beta,
                            "assistant_beta": a_beta,
                            "beta": a_beta,
                        }
                        # Override ASK_WINDOW via the policy kwargs
                        # _run_episode_impl passes ask_window from config
                        # We need to thread this through config_dict
                        cfg["ask_window_override"] = effective_window
                        m = _run_episode_impl(policy, seed, cfg)
                        rows.append({
                            "policy": policy, "K": K,
                            "obs_horizon": h,
                            "condition": cond_name,
                            "principal_beta": p_beta,
                            "assistant_beta": a_beta,
                            "rep_seed": int(rep_seed), "episode_id": int(eid),
                            "success": m["success"], "steps": m["steps"],
                            "questions_asked": m["questions_asked"],
                            "regret": m["regret"], "team_cost": m["team_cost"],
                        })
    fieldnames = ["policy", "K", "obs_horizon", "condition", "principal_beta",
                  "assistant_beta", "rep_seed", "episode_id",
                  "success", "steps", "questions_asked", "regret", "team_cost"]
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Obs hurts mismatch:", output_csv, f"({len(rows)} rows)")
    return rows
