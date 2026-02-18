"""
Generate comprehensive report from results/metrics.csv (full sweep results).
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.plots import load_metrics, aggregate_by_condition, plot_regret_vs_ambiguity, plot_questions_vs_ambiguity, plot_success_rate_vs_ambiguity
import numpy as np

CSV_PATH = "results/metrics.csv"
REPORT_MD = "results/full_report.md"


def generate_report():
    """Generate Markdown report from metrics.csv."""
    rows = load_metrics(CSV_PATH)
    if not rows:
        print(f"No metrics found at {CSV_PATH}")
        return

    agg = aggregate_by_condition(rows)
    K_vals = sorted(set(r["ambiguity_K"] for r in rows))
    eps_vals = sorted(set(r["eps"] for r in rows))
    beta_vals = sorted(set(r["beta"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))

    lines = [
        "# AskOrAct — Full Evaluation Report",
        "",
        "**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
        "## Setup",
        "",
        "- **Principal cannot pick objects** (`PRINCIPAL_CAN_PICK = False`).",
        "- **Success requires assistant to pick the true goal object.**",
        "- **Assistant picks whenever on any object cell** (may pick wrong object).",
        "- **Policies:** AskOrAct (asks when CostAsk < CostAct), NeverAsk (always act), AlwaysAsk (ask until entropy low).",
        "- **Conditions:** Ambiguity K ∈ {1, 2, 3, 4}, eps ∈ {0.0, 0.05, 0.1}, beta ∈ {1.0, 2.0, 4.0}.",
        "- **Episodes per condition:** 20.",
        "",
        "---",
        "",
        "## Summary by condition (K, eps, beta, policy)",
        "",
        "| K | eps | beta | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |",
        "|---|-----|------|--------|--------------|-----------|---------------|------------|-------------|",
    ]

    for K in K_vals:
        for eps in eps_vals:
            for beta in beta_vals:
                for policy in policies:
                    key = (K, eps, beta, policy)
                    if key not in agg:
                        continue
                    s = agg[key]
                    map_rate = s.get("map_correct_rate", 0.0)
                    lines.append(
                        "| {} | {} | {} | {} | {:.2%} | {:.1f} | {:.2f} | {:.1f} | {:.2%} |".format(
                            K, eps, beta, policy, s["success_rate"], s["avg_steps"],
                            s["avg_questions"], s["avg_regret"], map_rate
                        )
                    )

    lines.extend([
        "",
        "---",
        "",
        "## Aggregate by ambiguity K (averaged over eps and beta)",
        "",
        "| K | Policy | Success rate | Avg steps | Avg questions | Avg regret |",
        "|---|--------|--------------|-----------|---------------|------------|",
    ])

    for K in K_vals:
        for policy in policies:
            sub = [agg[k] for k in agg if k[0] == K and k[3] == policy]
            if not sub:
                continue
            success_rate = np.mean([s["success_rate"] for s in sub])
            avg_steps = np.mean([s["avg_steps"] for s in sub])
            avg_questions = np.mean([s["avg_questions"] for s in sub])
            avg_regret = np.mean([s["avg_regret"] for s in sub])
            lines.append(
                "| {} | {} | {:.2%} | {:.1f} | {:.2f} | {:.1f} |".format(
                    K, policy, success_rate, avg_steps, avg_questions, avg_regret
                )
            )

    lines.extend([
        "",
        "---",
        "",
        "## Key findings",
        "",
    ])

    # Compare policies at high ambiguity
    for K in [3, 4]:
        for policy in policies:
            sub = [agg[k] for k in agg if k[0] == K and k[3] == policy]
            if sub:
                success_rate = np.mean([s["success_rate"] for s in sub])
                avg_regret = np.mean([s["avg_regret"] for s in sub])
                avg_q = np.mean([s["avg_questions"] for s in sub])
                lines.append(f"- **K={K}, {policy}:** Success rate {success_rate:.2%}, avg regret {avg_regret:.1f}, avg questions {avg_q:.2f}")

    lines.extend([
        "",
        "---",
        "",
        "*Raw data: `results/metrics.csv`*",
        "*Plots: `results/regret_vs_ambiguity.png`, `results/questions_vs_ambiguity.png`, `results/success_rate_vs_ambiguity.png`*",
        "",
    ])

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_MD}")


def main():
    print("Generating report from", CSV_PATH)
    generate_report()
    print("Generating plots...")
    plot_regret_vs_ambiguity(CSV_PATH, "results/regret_vs_ambiguity.png")
    plot_questions_vs_ambiguity(CSV_PATH, "results/questions_vs_ambiguity.png")
    plot_success_rate_vs_ambiguity(CSV_PATH, "results/success_rate_vs_ambiguity.png")
    print("Done. Report:", REPORT_MD)


if __name__ == "__main__":
    main()
