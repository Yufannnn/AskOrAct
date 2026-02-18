"""
Milestone Feb 13 + Feb 20: gridworld, ambiguous instructions, scripted principal,
posterior inference, act-only assistant. No questions.

  Run one demo:     python scripts/run_milestone_feb13_feb20.py
  Run & report:     python scripts/run_milestone_feb13_feb20.py --report
"""

import sys
import os
import csv
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import config
from src.world import generate_world, instruction_to_candidate_goals
from src.inference import init_posterior, update_posterior
from src.agents import sample_principal_action, policy_never_ask

SEED = 42
MAX_STEPS_DEMO = 60

# Report config
REPORT_AMBIGUITY_LEVELS = [1, 2, 3, 4]
REPORT_EPS_LEVELS = [0.0, 0.05, 0.1]
REPORT_N_EPISODES = 15
REPORT_BASE_SEED = 100
RESULTS_DIR = "results"
METRICS_CSV = "results/milestone_feb13_feb20_metrics.csv"
SUMMARY_JSON = "results/milestone_feb13_feb20_summary.json"
REPORT_MD = "results/milestone_feb13_feb20_report.md"


def run_one_episode(seed, ambiguity_K, eps, beta, N, M):
    """Run a single Feb13+Feb20 episode (act-only). Returns dict of metrics."""
    rng = np.random.default_rng(seed)
    env, instruction_u, true_goal_obj_id = generate_world(seed, N=N, M=M, ambiguity_K=ambiguity_K)
    candidate_goals = instruction_to_candidate_goals(instruction_u, env)
    if not candidate_goals:
        return {"success": False, "steps": config.MAX_STEPS, "map_correct_final": False, "ambiguity_K": ambiguity_K, "eps": eps, "seed": seed}

    posterior = init_posterior(candidate_goals)
    principal_action_history = []
    steps = 0
    success = False

    while steps < config.MAX_STEPS:
        state = env.get_state()
        principal_action = sample_principal_action(state, true_goal_obj_id, env, rng, beta, eps)
        principal_action_history.append(principal_action)

        assistant_action, posterior = policy_never_ask(
            env=env, state=state, instruction_u=instruction_u, posterior=posterior,
            candidate_goals=candidate_goals, principal_action_history=principal_action_history,
            rng=rng, beta=beta, eps=eps, answer_noise=config.ANSWER_NOISE,
        )

        update_posterior(posterior, state, principal_action, candidate_goals, env, beta, eps)
        state, done, info = env.step(principal_action, assistant_action, true_goal_obj_id=true_goal_obj_id)
        steps += 1
        success = info.get("assistant_picked_goal", False)
        if done or success:
            break

    map_goal = max(candidate_goals, key=lambda g: posterior.get(g, 0)) if candidate_goals else None
    map_correct_final = (map_goal == true_goal_obj_id) if map_goal is not None else False

    return {
        "success": success,
        "steps": steps,
        "map_correct_final": map_correct_final,
        "ambiguity_K": ambiguity_K,
        "eps": eps,
        "beta": beta,
        "seed": seed,
        "n_candidates": len(candidate_goals),
    }


def run_and_collect():
    """Run episodes across conditions; return list of metric dicts and summary dict."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    beta = config.DEFAULT_BETA

    for K in REPORT_AMBIGUITY_LEVELS:
        for eps in REPORT_EPS_LEVELS:
            for ep in range(REPORT_N_EPISODES):
                seed = REPORT_BASE_SEED + K * 1000 + int(eps * 100) + ep
                m = run_one_episode(seed, ambiguity_K=K, eps=eps, beta=beta, N=config.DEFAULT_N, M=config.DEFAULT_M)
                rows.append(m)

    # Summary per (K, eps)
    summary = {}
    for K in REPORT_AMBIGUITY_LEVELS:
        for eps in REPORT_EPS_LEVELS:
            sub = [r for r in rows if r["ambiguity_K"] == K and r["eps"] == eps]
            if not sub:
                continue
            summary[(K, eps)] = {
                "success_rate": np.mean([r["success"] for r in sub]),
                "avg_steps": np.mean([r["steps"] for r in sub]),
                "map_correct_rate": np.mean([r["map_correct_final"] for r in sub]),
                "n_episodes": len(sub),
            }
    return rows, summary


def write_metrics(rows, summary):
    """Write CSV and JSON."""
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ambiguity_K", "eps", "beta", "seed", "success", "steps", "map_correct_final", "n_candidates"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "ambiguity_K": r["ambiguity_K"], "eps": r["eps"], "beta": r["beta"], "seed": r["seed"],
                "success": r["success"], "steps": r["steps"], "map_correct_final": r["map_correct_final"], "n_candidates": r["n_candidates"],
            })

    with open(SUMMARY_JSON, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    print("Wrote", METRICS_CSV)
    print("Wrote", SUMMARY_JSON)


def generate_report(rows, summary):
    """Write Markdown report to REPORT_MD."""
    lines = [
        "# Milestone Feb 13 + Feb 20 — Results Report",
        "",
        "**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
        "## Setup",
        "",
        "- **Feb 13:** Gridworld tasks, goal sets, ambiguous instruction templates, scripted (approximately rational) principal.",
        "- **Feb 20:** Posterior inference over goals from instruction + observed principal actions; act-only assistant (no questions).",
        "- **Policy:** Assistant always acts toward current MAP goal; posterior updated each step from principal action.",
        "- **Conditions:** Ambiguity K ∈ {1, 2, 3, 4} (number of candidate goals matching instruction), principal noise eps ∈ {0.0, 0.05, 0.1}.",
        "- **Episodes per condition:** " + str(REPORT_N_EPISODES) + ".",
        "",
        "---",
        "",
        "## Summary by condition",
        "",
        "| Ambiguity K | eps | Success rate | Avg steps | MAP correct (final) | N episodes |",
        "|-------------|-----|--------------|-----------|----------------------|------------|",
    ]

    for K in REPORT_AMBIGUITY_LEVELS:
        for eps in REPORT_EPS_LEVELS:
            key = (K, eps)
            if key not in summary:
                continue
            s = summary[key]
            lines.append(
                "| {} | {} | {:.2%} | {:.1f} | {:.2%} | {} |".format(
                    K, eps, s["success_rate"], s["avg_steps"], s["map_correct_rate"], s["n_episodes"]
                )
            )

    lines.extend([
        "",
        "---",
        "",
        "## Aggregate (all conditions)",
        "",
    ])

    all_success = [r["success"] for r in rows]
    all_steps = [r["steps"] for r in rows]
    all_map = [r["map_correct_final"] for r in rows]
    lines.append("- **Overall success rate:** {:.2%} ({}/{})".format(np.mean(all_success), sum(all_success), len(rows)))
    lines.append("- **Overall average steps (when run to end):** {:.1f}".format(np.mean(all_steps)))
    lines.append("- **Final MAP goal correct (when run ended):** {:.2%}".format(np.mean(all_map)))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Metrics and raw rows are in `milestone_feb13_feb20_metrics.csv` and `milestone_feb13_feb20_summary.json`.*")
    lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote", REPORT_MD)


def demo_only():
    """Single interactive demo (no report)."""
    rng = np.random.default_rng(SEED)
    env, instruction_u, true_goal_obj_id = generate_world(SEED, N=9, M=6, ambiguity_K=2)
    candidate_goals = instruction_to_candidate_goals(instruction_u, env)
    posterior = init_posterior(candidate_goals)
    principal_action_history = []

    print("=== Milestone Feb 13 + Feb 20 demo ===\n")
    print("Instruction (ambiguous):", instruction_u)
    print("True goal obj_id:", true_goal_obj_id)
    print("Candidate goals (from instruction):", candidate_goals)
    print()

    for step in range(MAX_STEPS_DEMO):
        state = env.get_state()
        print("--- Step", step + 1, "---")
        print(env.render_ascii(true_goal_obj_id))
        print("Posterior over goals:", {g: round(posterior.get(g, 0), 3) for g in candidate_goals})

        principal_action = sample_principal_action(
            state, true_goal_obj_id, env, rng, config.DEFAULT_BETA, config.DEFAULT_EPS
        )
        principal_action_history.append(principal_action)

        assistant_action, posterior = policy_never_ask(
            env=env, state=state, instruction_u=instruction_u, posterior=posterior,
            candidate_goals=candidate_goals, principal_action_history=principal_action_history,
            rng=rng, beta=config.DEFAULT_BETA, eps=config.DEFAULT_EPS, answer_noise=config.ANSWER_NOISE,
        )
        print("Principal action:", principal_action, "| Assistant (act only):", assistant_action)

        update_posterior(posterior, state, principal_action, candidate_goals, env, config.DEFAULT_BETA, config.DEFAULT_EPS)
        state, done, info = env.step(principal_action, assistant_action, true_goal_obj_id=true_goal_obj_id)
        if info.get("assistant_picked_goal", False):
            print("Goal satisfied at step", step + 1, "(assistant picked true goal)")
            break
        print()

    if not info.get("assistant_picked_goal", False):
        print("Max steps reached; goal not satisfied (only assistant pick counts).")
    print("Final state:")
    print(env.render_ascii(true_goal_obj_id))


def main():
    parser = argparse.ArgumentParser(description="Feb 13 + Feb 20 milestone: run demo or run experiments and generate report.")
    parser.add_argument("--report", action="store_true", help="Run episodes, save metrics, and generate Markdown report.")
    args = parser.parse_args()

    if args.report:
        print("Running Feb 13 + Feb 20 experiments (act-only assistant)...")
        rows, summary = run_and_collect()
        write_metrics(rows, summary)
        generate_report(rows, summary)
        print("Done. Report:", REPORT_MD)
    else:
        demo_only()


if __name__ == "__main__":
    main()
