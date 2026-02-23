"""Generate comprehensive report from results/metrics.csv (full sweep results)."""

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.plots import (  # noqa: E402
    load_metrics,
    load_ablation_metrics,
    aggregate_by_condition,
    plot_main_dashboard,
    plot_ablations_dashboard,
    bootstrap_mean_ci,
)
from src import config  # noqa: E402

CSV_PATH = "results/metrics.csv"
ABLATION_CSV_PATH = "results/metrics_ablations.csv"
REPORT_MD = "results/full_report.md"


def generate_report():
    """Generate Markdown report from metrics.csv."""
    rows = load_metrics(CSV_PATH)
    if not rows:
        print(f"No metrics found at {CSV_PATH}")
        return

    agg = aggregate_by_condition(rows)
    k_vals = sorted(set(r["ambiguity_K"] for r in rows))
    eps_vals = sorted(set(r["eps"] for r in rows))
    beta_vals = sorted(set(r["beta"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    rep_vals = sorted(set(r.get("rep_seed", 0) for r in rows))
    episodes_per_seed = int(getattr(config, "N_EPISODES_PER_SEED", getattr(config, "N_EPISODES_PER_CONDITION", 20)))
    total_per_condition = len(rep_vals) * episodes_per_seed

    lines = [
        "# AskOrAct - Full Evaluation Report",
        "",
        "**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
        "## Setup",
        "",
        "- **Principal cannot pick objects** (`PRINCIPAL_CAN_PICK = False`).",
        "- **Success requires assistant to pick the true goal object.**",
        "- **Episode deadline:** per-episode max steps = oracle shortest assistant path + margin.",
        "- **Cost rule:** `team_cost = steps + QUESTION_COST * questions_asked`; failed episodes use `team_cost = episode_max_steps + QUESTION_COST * questions_asked`.",
        "- **Policies:** AskOrAct, NeverAsk, AlwaysAsk.",
        "- **Conditions:** Ambiguity K in {1, 2, 3, 4}, eps in {0.0, 0.05, 0.1}, beta in {1.0, 2.0, 4.0}.",
        f"- **Replicate seeds:** {rep_vals}.",
        f"- **Episodes per (policy, K, eps, beta):** {total_per_condition} ({len(rep_vals)} reps x {episodes_per_seed} eps/rep).",
        "- **Uncertainty:** 95% bootstrap CI over episodes (B=1000).",
        "",
        "---",
        "",
        "## Summary by condition (K, eps, beta, policy)",
        "",
        "| K | eps | beta | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |",
        "|---|-----|------|--------|--------------|-----------|---------------|------------|-------------|",
    ]

    for K in k_vals:
        for eps in eps_vals:
            for beta in beta_vals:
                for policy in policies:
                    key = (K, eps, beta, policy)
                    if key not in agg:
                        continue
                    s = agg[key]
                    lines.append(
                        "| {} | {} | {} | {} | {:.2%} | {:.1f} | {:.2f} | {:.2f} | {:.2%} |".format(
                            K,
                            eps,
                            beta,
                            policy,
                            s["success_rate"],
                            s["avg_steps"],
                            s["avg_questions"],
                            s["avg_regret"],
                            s.get("map_correct_rate", 0.0),
                        )
                    )

    lines.extend([
        "",
        "---",
        "",
        "## Aggregate by K (averaged over eps and beta)",
        "",
        "| K | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |",
        "|---|--------|--------------|-----------|---------------|------------|-------------|",
    ])

    for K in k_vals:
        for policy in policies:
            sub = [agg[k] for k in agg if k[0] == K and k[3] == policy]
            if not sub:
                continue
            lines.append(
                "| {} | {} | {:.2%} | {:.1f} | {:.2f} | {:.2f} | {:.2%} |".format(
                    K,
                    policy,
                    float(np.mean([s["success_rate"] for s in sub])),
                    float(np.mean([s["avg_steps"] for s in sub])),
                    float(np.mean([s["avg_questions"] for s in sub])),
                    float(np.mean([s["avg_regret"] for s in sub])),
                    float(np.mean([s.get("map_correct_rate", 0.0) for s in sub])),
                )
            )

    lines.extend([
        "",
        "---",
        "",
        "## High-Ambiguity Snapshot (K=3,4)",
        "",
    ])

    for policy in policies:
        sub = [agg[k] for k in agg if k[0] in (3, 4) and k[3] == policy]
        if not sub:
            continue
        lines.append(
            "- **{}:** success {:.2%}, avg regret {:.2f}, avg questions {:.2f}.".format(
                policy,
                float(np.mean([s["success_rate"] for s in sub])),
                float(np.mean([s["avg_regret"] for s in sub])),
                float(np.mean([s["avg_questions"] for s in sub])),
            )
        )

    lines.extend([
        "",
        "## Robustness Summary (K=3,4)",
        "",
        "All results are averaged over replicate seeds; uncertainty is 95% bootstrap CI over episodes.",
        "",
        "| K | Policy | Success (mean [95% CI]) | Regret (mean [95% CI]) | Questions (mean [95% CI]) |",
        "|---|--------|--------------------------|-------------------------|----------------------------|",
    ])

    for K in [3, 4]:
        for policy in policies:
            sub = [r for r in rows if r["ambiguity_K"] == K and r["policy"] == policy]
            if not sub:
                continue
            success_vals = [1.0 if r["success"] else 0.0 for r in sub]
            regret_vals = [float(r["regret"]) for r in sub]
            q_vals = [float(r["questions_asked"]) for r in sub]

            s_mean, s_lo, s_hi = bootstrap_mean_ci(success_vals, n_boot=1000, rng_seed=10 + K)
            r_mean, r_lo, r_hi = bootstrap_mean_ci(regret_vals, n_boot=1000, rng_seed=20 + K)
            q_mean, q_lo, q_hi = bootstrap_mean_ci(q_vals, n_boot=1000, rng_seed=30 + K)

            lines.append(
                "| {} | {} | {:.2%} [{:.2%}, {:.2%}] | {:.2f} [{:.2f}, {:.2f}] | {:.2f} [{:.2f}, {:.2f}] |".format(
                    K,
                    policy,
                    s_mean,
                    s_lo,
                    s_hi,
                    r_mean,
                    r_lo,
                    r_hi,
                    q_mean,
                    q_lo,
                    q_hi,
                )
            )

    def _paired_delta_ci(metric_name, policy_a, policy_b, k_set, rng_seed):
        """
        Bootstrap CI for delta of means (policy_a - policy_b), computed on paired keys:
        (K, eps, beta, rep_seed, episode_id).
        """
        def _metric(row):
            if metric_name == "success":
                return 1.0 if row["success"] else 0.0
            if metric_name == "regret":
                return float(row["regret"])
            raise ValueError(metric_name)

        rows_a = [
            r for r in rows
            if r["policy"] == policy_a and r["ambiguity_K"] in k_set
        ]
        rows_b = [
            r for r in rows
            if r["policy"] == policy_b and r["ambiguity_K"] in k_set
        ]
        map_a = {
            (r["ambiguity_K"], r["eps"], r["beta"], r.get("rep_seed", 0), r.get("episode_id", 0)): _metric(r)
            for r in rows_a
        }
        map_b = {
            (r["ambiguity_K"], r["eps"], r["beta"], r.get("rep_seed", 0), r.get("episode_id", 0)): _metric(r)
            for r in rows_b
        }
        common = sorted(set(map_a.keys()) & set(map_b.keys()))
        if not common:
            return np.nan, np.nan, np.nan, 0
        deltas = [map_a[k] - map_b[k] for k in common]
        mean, lo, hi = bootstrap_mean_ci(deltas, n_boot=1000, rng_seed=rng_seed)
        return mean, lo, hi, len(deltas)

    lines.extend([
        "",
        "## Policy Difference CIs (K=3,4)",
        "",
        "Deltas are bootstrap-estimated on paired episode keys: (K, eps, beta, rep_seed, episode_id).",
        "",
        "| Contrast | Delta mean [95% CI] | N paired episodes |",
        "|----------|-----------------------|-------------------|",
    ])

    ds_mean, ds_lo, ds_hi, ds_n = _paired_delta_ci("success", "ask_or_act", "never_ask", {3, 4}, 2001)
    lines.append(
        "| DeltaSuccess (AskOrAct - NeverAsk) | {:.2%} [{:.2%}, {:.2%}] | {} |".format(
            ds_mean, ds_lo, ds_hi, ds_n
        )
    )

    dr_mean, dr_lo, dr_hi, dr_n = _paired_delta_ci("regret", "ask_or_act", "always_ask", {3, 4}, 2002)
    lines.append(
        "| DeltaRegret (AskOrAct - AlwaysAsk) | {:.2f} [{:.2f}, {:.2f}] | {} |".format(
            dr_mean, dr_lo, dr_hi, dr_n
        )
    )

    lines.extend([
        "",
        "## Failure Mode Breakdown",
        "",
        "| K | Policy | Failure rate | failure_by_timeout rate | failure_by_wrong_pick rate |",
        "|---|--------|--------------|-------------------------|----------------------------|",
    ])

    for K in [3, 4]:
        for policy in policies:
            sub = [r for r in rows if r["ambiguity_K"] == K and r["policy"] == policy]
            if not sub:
                continue
            fail_rate = float(np.mean([0.0 if r["success"] else 1.0 for r in sub]))
            timeout_rate = float(np.mean([1.0 if r.get("failure_by_timeout", False) else 0.0 for r in sub]))
            wrongpick_rate = float(np.mean([1.0 if r.get("failure_by_wrong_pick", False) else 0.0 for r in sub]))
            lines.append(
                "| {} | {} | {:.2%} | {:.2%} | {:.2%} |".format(
                    K, policy, fail_rate, timeout_rate, wrongpick_rate
                )
            )

    wrong_pick_rows = [r for r in rows if r.get("terminated_by_wrong_pick", False)]
    lines.extend([
        "",
        "## Wrong-Pick Fail Semantics",
        "",
        "- Main sweep keeps `WRONG_PICK_FAIL = False` for comparability." if not wrong_pick_rows else "- Wrong-pick fail was active and triggered in this run.",
        "",
    ])

    ablation_rows = load_ablation_metrics(ABLATION_CSV_PATH)
    if ablation_rows:
        lines.extend([
            "## Ablation Notes",
            "",
            "- **Mode A (time-only):** ASK counts as step, QUESTION_COST = 0.0.",
            "- **Mode B (comm-cost-only):** ASK does not count as step, QUESTION_COST = 0.5.",
            "- **Mode C (both):** ASK counts as step, QUESTION_COST = 0.5.",
            "- This separates temporal cost from communication cost.",
            "",
        ])

        def _mode_slice(mode_name):
            return [r for r in ablation_rows if r["mode_name"] == mode_name and not r["wrong_pick_fail"] and r["K"] in (3, 4)]

        for mode_name in sorted(set(r["mode_name"] for r in ablation_rows)):
            sub = _mode_slice(mode_name)
            if not sub:
                continue
            lines.append(f"- **{mode_name} (K=3,4, wrong_pick_fail=False):**")
            for policy in ["ask_or_act", "never_ask", "always_ask"]:
                psub = [r for r in sub if r["policy"] == policy]
                if not psub:
                    continue
                lines.append(
                    "  {} -> success {:.2%}, regret {:.2f}, questions {:.2f}".format(
                        policy,
                        float(np.mean([r["success"] for r in psub])),
                        float(np.mean([r["regret"] for r in psub])),
                        float(np.mean([r["questions_asked"] for r in psub])),
                    )
                )

    lines.extend([
        "",
        "---",
        "",
        "*Raw data: `results/metrics.csv`*",
        "*Main plots: `results/main_dashboard.png`*",
        "*Ablations: `results/metrics_ablations.csv` and `results/ablations_dashboard.png`*",
        "",
    ])

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_MD}")


def main():
    print("Generating report from", CSV_PATH)
    generate_report()
    print("Generating plots...")
    plot_main_dashboard(CSV_PATH, "results/main_dashboard.png")
    if os.path.exists(ABLATION_CSV_PATH):
        plot_ablations_dashboard(ABLATION_CSV_PATH, "results/ablations_dashboard.png")
    print("Done. Report:", REPORT_MD)


if __name__ == "__main__":
    main()

