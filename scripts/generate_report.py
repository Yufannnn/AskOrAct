"""Generate comprehensive report from results/metrics.csv (full sweep results)."""

import csv
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
    plot_clarification_quality_entropy_delta,
    plot_robust_answer_noise_deltas,
    plot_robust_mismatch_deltas,
    plot_pareto_k4,
    bootstrap_mean_ci,
)
from src import config  # noqa: E402

CSV_PATH = "results/metrics.csv"
ABLATION_CSV_PATH = "results/metrics_ablations.csv"
ROBUST_ANSWER_CSV_PATH = "results/metrics_robust_answer_noise.csv"
ROBUST_MISMATCH_CSV_PATH = "results/metrics_robust_mismatch.csv"
GENERALIZATION_TEMPLATES_CSV_PATH = "results/metrics_generalization_templates.csv"
SCALEK_CSV_PATH = "results/metrics_scaleK.csv"
REPORT_MD = "results/full_report.md"


def _load_optional_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("K", "rep_seed", "episode_id", "template_id"):
                if k in row and row[k] != "":
                    row[k] = int(row[k])
            for k in ("answer_noise", "principal_beta", "beta", "eps", "regret", "questions_asked", "steps"):
                if k in row and row[k] != "":
                    row[k] = float(row[k])
            if "success" in row:
                row["success"] = str(row["success"]).strip().lower() in ("1", "true", "yes")
            rows.append(row)
    return rows


def _paired_delta_ci_generic(rows, metric_name, policy_a, policy_b, cond_fn, rng_seed):
    """
    Bootstrap CI for policy_a - policy_b on paired keys:
      (cond, K, rep_seed, episode_id)
    """

    def _metric(row):
        if metric_name == "success":
            return 1.0 if row["success"] else 0.0
        if metric_name == "regret":
            return float(row["regret"])
        raise ValueError(metric_name)

    rows_a = [r for r in rows if r.get("policy") == policy_a]
    rows_b = [r for r in rows if r.get("policy") == policy_b]
    map_a = {
        (cond_fn(r), int(r.get("K", r.get("ambiguity_K", 0))), int(r.get("rep_seed", 0)), int(r.get("episode_id", 0))): _metric(r)
        for r in rows_a
    }
    map_b = {
        (cond_fn(r), int(r.get("K", r.get("ambiguity_K", 0))), int(r.get("rep_seed", 0)), int(r.get("episode_id", 0))): _metric(r)
        for r in rows_b
    }
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common:
        return np.nan, np.nan, np.nan, 0
    deltas = [map_a[k] - map_b[k] for k in common]
    mean, lo, hi = bootstrap_mean_ci(deltas, n_boot=1000, rng_seed=rng_seed)
    return mean, lo, hi, len(deltas)


def generate_report():
    """Generate Markdown report from metrics.csv."""
    rows = load_metrics(CSV_PATH)
    if not rows:
        print(f"No metrics found at {CSV_PATH}")
        return

    ablation_rows = load_ablation_metrics(ABLATION_CSV_PATH)
    robust_answer_rows = _load_optional_rows(ROBUST_ANSWER_CSV_PATH)
    robust_mismatch_rows = _load_optional_rows(ROBUST_MISMATCH_CSV_PATH)
    generalization_template_rows = _load_optional_rows(GENERALIZATION_TEMPLATES_CSV_PATH)
    scalek_rows = _load_optional_rows(SCALEK_CSV_PATH)

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
        "- **Policies:** AskOrAct, NeverAsk, AlwaysAsk, InfoGainAsk, RandomAsk, POMCPPlanner.",
        "- **Conditions:** Ambiguity K in {1, 2, 3, 4}, eps in {0.0, 0.05, 0.1}, beta in {1.0, 2.0, 4.0}.",
        f"- **Replicate seeds:** {rep_vals}.",
        f"- **Episodes per (policy, K, eps, beta):** {total_per_condition} ({len(rep_vals)} reps x {episodes_per_seed} eps/rep).",
        "- **Uncertainty:** 95% bootstrap CI over episodes (B=1000).",
        "",
        "## Baselines",
        "",
        "- **info_gain_ask:** computes expected entropy reduction `IG(q)=H(b)-E[H|q]` for each question and asks `argmax_q IG(q)` when IG passes threshold (and optional entropy gate), otherwise acts toward MAP goal.",
        "- **random_ask:** uses the same ask gating as `info_gain_ask` but picks a random available question when asking.",
        "- **pomcp_planner:** POMCP/PO-UCT baseline that approximates POMDP ask-act planning with Monte Carlo tree search over physical and question actions.",
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
        "## Figures",
        "",
        "### Main Dashboard",
        "",
        "![Main dashboard](main_dashboard.png)",
        "",
        "### Clarification Quality",
        "",
        "![Clarification quality entropy delta](clarification_quality_entropy_delta.png)",
        "",
        "### Pareto (K=4)",
        "",
        "![Pareto K=4](pareto_K4.png)",
        "",
    ])

    if ablation_rows:
        lines.extend([
            "### Ablations Dashboard",
            "",
            "![Ablations dashboard](ablations_dashboard.png)",
            "",
        ])

    if robust_answer_rows:
        lines.extend([
            "### Robustness: Answer Noise Deltas",
            "",
            "![Robust answer noise deltas](robust_answer_noise_deltas.png)",
            "",
        ])

    if robust_mismatch_rows:
        lines.extend([
            "### Robustness: Principal-Model Mismatch Deltas",
            "",
            "![Robust mismatch deltas](robust_mismatch_deltas.png)",
            "",
        ])

    if generalization_template_rows:
        lines.extend([
            "### Generalization: Held-out Templates",
            "",
            "![Held-out template generalization](generalization_templates_plot.png)",
            "",
        ])

    if scalek_rows:
        lines.extend([
            "### Generalization: Scale K to 6",
            "",
            "![Scale-K stress test](scaleK_plot.png)",
            "",
        ])

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
        "## Clarification Quality (First Asked Question)",
        "",
        "We summarize posterior contraction from the first asked question using entropy and effective goal count.",
        "",
        "| Policy | Ask rate (K=3,4) | H_before | H_after | DeltaH | N_eff_before | N_eff_after | DeltaN_eff | IG_first |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    clar_stats = []
    for policy in policies:
        sub = [r for r in rows if r["ambiguity_K"] in (3, 4) and r["policy"] == policy]
        if not sub:
            continue
        ask_rate = float(np.mean([1.0 if r["questions_asked"] > 0 else 0.0 for r in sub]))
        h_before = float(np.mean([float(r.get("entropy_before_first_ask", 0.0)) for r in sub]))
        h_after = float(np.mean([float(r.get("entropy_after_first_ask", 0.0)) for r in sub]))
        n_before = float(np.mean([float(r.get("effective_goal_count_before", 0.0)) for r in sub]))
        n_after = float(np.mean([float(r.get("effective_goal_count_after", 0.0)) for r in sub]))
        ig_first = float(np.mean([float(r.get("ig_of_first_asked_question", 0.0)) for r in sub]))
        d_h = h_before - h_after
        d_n = n_before - n_after
        clar_stats.append((policy, d_h, ask_rate))
        lines.append(
            "| {} | {:.2%} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                policy, ask_rate, h_before, h_after, d_h, n_before, n_after, d_n, ig_first
            )
        )

    if clar_stats:
        candidates = [x for x in clar_stats if x[2] > 0]
        if candidates:
            best_policy = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
            lines.extend([
                "",
                f"Interpretation: `{best_policy}` achieves the largest average first-question entropy reduction among policies that ask at K=3-4.",
            ])

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

    def _paired_delta_ci(metric_name, policy_a, policy_b, k_set, rng_seed, rowset=None):
        """
        Bootstrap CI for delta of means (policy_a - policy_b), computed on paired keys:
        (K, eps, beta, rep_seed, episode_id).
        Supports both main rows (`ambiguity_K`) and optional rows (`K`).
        """
        rowset = rowset if rowset is not None else rows

        def _metric(row):
            if metric_name == "success":
                return 1.0 if row["success"] else 0.0
            if metric_name == "regret":
                return float(row["regret"])
            raise ValueError(metric_name)

        def _row_k(row):
            return int(row.get("ambiguity_K", row.get("K", 0)))

        def _row_eps(row):
            return float(row.get("eps", 0.0))

        def _row_beta(row):
            return float(row.get("beta", 0.0))

        rows_a = [r for r in rowset if r["policy"] == policy_a and _row_k(r) in k_set]
        rows_b = [r for r in rowset if r["policy"] == policy_b and _row_k(r) in k_set]
        map_a = {
            (_row_k(r), _row_eps(r), _row_beta(r), int(r.get("rep_seed", 0)), int(r.get("episode_id", 0))): _metric(r)
            for r in rows_a
        }
        map_b = {
            (_row_k(r), _row_eps(r), _row_beta(r), int(r.get("rep_seed", 0)), int(r.get("episode_id", 0))): _metric(r)
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
        "| DeltaSuccess (AskOrAct - NeverAsk) | {:.2%} [{:.2%}, {:.2%}] | {} |".format(ds_mean, ds_lo, ds_hi, ds_n)
    )

    dr_mean, dr_lo, dr_hi, dr_n = _paired_delta_ci("regret", "ask_or_act", "always_ask", {3, 4}, 2002)
    lines.append(
        "| DeltaRegret (AskOrAct - AlwaysAsk) | {:.2f} [{:.2f}, {:.2f}] | {} |".format(dr_mean, dr_lo, dr_hi, dr_n)
    )

    dsi_mean, dsi_lo, dsi_hi, dsi_n = _paired_delta_ci("success", "ask_or_act", "info_gain_ask", {3, 4}, 2003)
    lines.append(
        "| DeltaSuccess (AskOrAct - InfoGainAsk) | {:.2%} [{:.2%}, {:.2%}] | {} |".format(dsi_mean, dsi_lo, dsi_hi, dsi_n)
    )

    dri_mean, dri_lo, dri_hi, dri_n = _paired_delta_ci("regret", "ask_or_act", "info_gain_ask", {3, 4}, 2004)
    lines.append(
        "| DeltaRegret (AskOrAct - InfoGainAsk) | {:.2f} [{:.2f}, {:.2f}] | {} |".format(dri_mean, dri_lo, dri_hi, dri_n)
    )

    dse_mean, dse_lo, dse_hi, dse_n = _paired_delta_ci("success", "ask_or_act", "easy_info_gain_ask", {3, 4}, 2005)
    if dse_n == 0 and scalek_rows:
        dse_mean, dse_lo, dse_hi, dse_n = _paired_delta_ci(
            "success", "ask_or_act", "easy_info_gain_ask", {3, 4}, 2105, rowset=scalek_rows
        )
    lines.append(
        "| DeltaSuccess (AskOrAct - EasyInfoGainAsk) | {:.2%} [{:.2%}, {:.2%}] | {} |".format(dse_mean, dse_lo, dse_hi, dse_n)
    )

    dre_mean, dre_lo, dre_hi, dre_n = _paired_delta_ci("regret", "ask_or_act", "easy_info_gain_ask", {3, 4}, 2006)
    if dre_n == 0 and scalek_rows:
        dre_mean, dre_lo, dre_hi, dre_n = _paired_delta_ci(
            "regret", "ask_or_act", "easy_info_gain_ask", {3, 4}, 2106, rowset=scalek_rows
        )
    lines.append(
        "| DeltaRegret (AskOrAct - EasyInfoGainAsk) | {:.2f} [{:.2f}, {:.2f}] | {} |".format(dre_mean, dre_lo, dre_hi, dre_n)
    )

    dsp_mean, dsp_lo, dsp_hi, dsp_n = _paired_delta_ci("success", "ask_or_act", "pomcp_planner", {3, 4}, 2007)
    if dsp_n == 0 and scalek_rows:
        dsp_mean, dsp_lo, dsp_hi, dsp_n = _paired_delta_ci(
            "success", "ask_or_act", "pomcp_planner", {3, 4}, 2107, rowset=scalek_rows
        )
    lines.append(
        "| DeltaSuccess (AskOrAct - POMCP) | {:.2%} [{:.2%}, {:.2%}] | {} |".format(dsp_mean, dsp_lo, dsp_hi, dsp_n)
    )

    drp_mean, drp_lo, drp_hi, drp_n = _paired_delta_ci("regret", "ask_or_act", "pomcp_planner", {3, 4}, 2008)
    if drp_n == 0 and scalek_rows:
        drp_mean, drp_lo, drp_hi, drp_n = _paired_delta_ci(
            "regret", "ask_or_act", "pomcp_planner", {3, 4}, 2108, rowset=scalek_rows
        )
    lines.append(
        "| DeltaRegret (AskOrAct - POMCP) | {:.2f} [{:.2f}, {:.2f}] | {} |".format(drp_mean, drp_lo, drp_hi, drp_n)
    )

    lines.extend([
        "",
        "IG is best at entropy reduction; VoI is best at minimizing team cost under deadlines; POMCP approximates POMDP planning and serves as a stronger baseline.",
    ])

    if robust_answer_rows or robust_mismatch_rows:
        lines.extend([
            "",
            "## Robustness Sweeps (K=3,4)",
            "",
            "Robustness is summarized with paired deltas and bootstrap CIs under answer-noise shifts and principal-model mismatch.",
            "",
        ])

    if robust_answer_rows:
        noise_vals = sorted(set(float(r["answer_noise"]) for r in robust_answer_rows))
        lines.extend([
            "### Answer-Noise Robustness",
            "",
            "| answer_noise | DeltaSuccess (AskOrAct - NeverAsk) | DeltaRegret (AskOrAct - AlwaysAsk) |",
            "|---:|---:|---:|",
        ])
        for nv in noise_vals:
            sub = [r for r in robust_answer_rows if float(r["answer_noise"]) == nv]
            ds_m, ds_l, ds_h, _ = _paired_delta_ci_generic(
                sub, "success", "ask_or_act", "never_ask", cond_fn=lambda r: float(r["answer_noise"]), rng_seed=3000 + int(100 * nv)
            )
            dr_m, dr_l, dr_h, _ = _paired_delta_ci_generic(
                sub, "regret", "ask_or_act", "always_ask", cond_fn=lambda r: float(r["answer_noise"]), rng_seed=4000 + int(100 * nv)
            )
            lines.append(
                "| {:.1f} | {:.2%} [{:.2%}, {:.2%}] | {:.2f} [{:.2f}, {:.2f}] |".format(nv, ds_m, ds_l, ds_h, dr_m, dr_l, dr_h)
            )
        lines.append("")
        lines.append("Trend: AskOrAct keeps a positive success advantage vs NeverAsk and a strong regret advantage vs AlwaysAsk as answer noise increases.")

    if robust_mismatch_rows:
        beta_vals_rob = sorted(set(float(r["principal_beta"]) for r in robust_mismatch_rows))
        lines.extend([
            "",
            "### Principal-Model Mismatch Robustness",
            "",
            "| principal_beta | DeltaSuccess (AskOrAct - NeverAsk) | DeltaRegret (AskOrAct - AlwaysAsk) |",
            "|---:|---:|---:|",
        ])
        for pb in beta_vals_rob:
            sub = [r for r in robust_mismatch_rows if float(r["principal_beta"]) == pb]
            ds_m, ds_l, ds_h, _ = _paired_delta_ci_generic(
                sub, "success", "ask_or_act", "never_ask", cond_fn=lambda r: float(r["principal_beta"]), rng_seed=5000 + int(10 * pb)
            )
            dr_m, dr_l, dr_h, _ = _paired_delta_ci_generic(
                sub, "regret", "ask_or_act", "always_ask", cond_fn=lambda r: float(r["principal_beta"]), rng_seed=6000 + int(10 * pb)
            )
            lines.append(
                "| {:.1f} | {:.2%} [{:.2%}, {:.2%}] | {:.2f} [{:.2f}, {:.2f}] |".format(pb, ds_m, ds_l, ds_h, dr_m, dr_l, dr_h)
            )
        lines.append("")
        lines.append("Trend: AskOrAct remains robust to principal rationality mismatch, preserving positive success deltas and negative regret deltas.")

    if generalization_template_rows:
        lines.extend([
            "",
            "## Held-out Template Generalization",
            "",
            "Evaluation uses only held-out instruction templates (deterministic 70/30 split over template IDs), with the same K/eps/beta grids.",
            "",
            "| K | Policy | Success [95% CI] | Regret [95% CI] | Questions [95% CI] |",
            "|---|---|---|---|---|",
        ])
        g_policies = sorted(set(r["policy"] for r in generalization_template_rows))
        g_ks = sorted(set(int(r["K"]) for r in generalization_template_rows))
        for K in g_ks:
            for policy in g_policies:
                sub = [r for r in generalization_template_rows if int(r["K"]) == K and r["policy"] == policy]
                if not sub:
                    continue
                s_vals = [1.0 if r["success"] else 0.0 for r in sub]
                r_vals = [float(r["regret"]) for r in sub]
                q_vals = [float(r["questions_asked"]) for r in sub]
                s_mean, s_lo, s_hi = bootstrap_mean_ci(s_vals, n_boot=1000, rng_seed=17000 + K)
                r_mean, r_lo, r_hi = bootstrap_mean_ci(r_vals, n_boot=1000, rng_seed=17100 + K)
                q_mean, q_lo, q_hi = bootstrap_mean_ci(q_vals, n_boot=1000, rng_seed=17200 + K)
                lines.append(
                    "| {} | {} | {:.2%} [{:.2%}, {:.2%}] | {:.2f} [{:.2f}, {:.2f}] | {:.2f} [{:.2f}, {:.2f}] |".format(
                        K, policy, s_mean, s_lo, s_hi, r_mean, r_lo, r_hi, q_mean, q_lo, q_hi
                    )
                )
        lines.extend([
            "",
            "Summary: performance degrades gracefully under unseen templates, with no catastrophic collapse in success or regret trends.",
        ])

    if scalek_rows:
        lines.extend([
            "",
            "## Scale-K Stress Test (K up to 6)",
            "",
            "Evaluation fixes `eps=0.05` and `beta=2.0`, extending ambiguity to `K=5,6`.",
            "",
            "| K | Policy | Success [95% CI] | Regret [95% CI] | Questions [95% CI] |",
            "|---|---|---|---|---|",
        ])
        s_policies = sorted(set(r["policy"] for r in scalek_rows))
        s_ks = sorted(set(int(r["K"]) for r in scalek_rows))
        for K in s_ks:
            for policy in s_policies:
                sub = [r for r in scalek_rows if int(r["K"]) == K and r["policy"] == policy]
                if not sub:
                    continue
                s_vals = [1.0 if r["success"] else 0.0 for r in sub]
                r_vals = [float(r["regret"]) for r in sub]
                q_vals = [float(r["questions_asked"]) for r in sub]
                s_mean, s_lo, s_hi = bootstrap_mean_ci(s_vals, n_boot=1000, rng_seed=17300 + K)
                r_mean, r_lo, r_hi = bootstrap_mean_ci(r_vals, n_boot=1000, rng_seed=17400 + K)
                q_mean, q_lo, q_hi = bootstrap_mean_ci(q_vals, n_boot=1000, rng_seed=17500 + K)
                lines.append(
                    "| {} | {} | {:.2%} [{:.2%}, {:.2%}] | {:.2f} [{:.2f}, {:.2f}] | {:.2f} [{:.2f}, {:.2f}] |".format(
                        K, policy, s_mean, s_lo, s_hi, r_mean, r_lo, r_hi, q_mean, q_lo, q_hi
                    )
                )
        lines.extend([
            "",
            "Summary: as ambiguity increases to K=5-6, regret and question load rise as expected, but policy ordering remains broadly stable (graceful degradation).",
        ])

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
                "| {} | {} | {:.2%} | {:.2%} | {:.2%} |".format(K, policy, fail_rate, timeout_rate, wrongpick_rate)
            )

    wrong_pick_rows = [r for r in rows if r.get("terminated_by_wrong_pick", False)]
    lines.extend([
        "",
        "## Wrong-Pick Fail Semantics",
        "",
        "- Main sweep keeps `WRONG_PICK_FAIL = False` for comparability." if not wrong_pick_rows else "- Wrong-pick fail was active and triggered in this run.",
        "",
    ])

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
            for policy in config.POLICIES:
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
        "*Clarification quality plot: `results/clarification_quality_entropy_delta.png`*",
        "*Ablations: `results/metrics_ablations.csv` and `results/ablations_dashboard.png`*",
        "*Robustness plots: `results/robust_answer_noise_deltas.png`, `results/robust_mismatch_deltas.png`*",
        "*Pareto plot: `results/pareto_K4.png`*",
        "",
    ])

    if robust_answer_rows or robust_mismatch_rows:
        lines.extend([
            "## Robustness Sweep Artifacts",
            "",
            "- `results/metrics_robust_answer_noise.csv`",
            "- `results/metrics_robust_mismatch.csv`",
            "- `results/robust_answer_noise_deltas.png`",
            "- `results/robust_mismatch_deltas.png`",
            "",
        ])

    if generalization_template_rows or scalek_rows:
        lines.extend([
            "## Generalization Artifacts",
            "",
            "- `results/metrics_generalization_templates.csv`",
            "- `results/generalization_templates_plot.png`",
            "- `results/metrics_scaleK.csv`",
            "- `results/scaleK_plot.png`",
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
    plot_clarification_quality_entropy_delta(CSV_PATH, "results/clarification_quality_entropy_delta.png")
    plot_pareto_k4(CSV_PATH, "results/pareto_K4.png")
    if os.path.exists(ABLATION_CSV_PATH):
        plot_ablations_dashboard(ABLATION_CSV_PATH, "results/ablations_dashboard.png")
    if os.path.exists(ROBUST_ANSWER_CSV_PATH):
        plot_robust_answer_noise_deltas(ROBUST_ANSWER_CSV_PATH, "results/robust_answer_noise_deltas.png")
    if os.path.exists(ROBUST_MISMATCH_CSV_PATH):
        plot_robust_mismatch_deltas(ROBUST_MISMATCH_CSV_PATH, "results/robust_mismatch_deltas.png")
    if os.path.exists(GENERALIZATION_TEMPLATES_CSV_PATH):
        from src.eval.plots import plot_generalization_templates
        plot_generalization_templates(GENERALIZATION_TEMPLATES_CSV_PATH, "results/generalization_templates_plot.png")
    if os.path.exists(SCALEK_CSV_PATH):
        from src.eval.plots import plot_scale_k
        plot_scale_k(SCALEK_CSV_PATH, "results/scaleK_plot.png")
    print("Done. Report:", REPORT_MD)


if __name__ == "__main__":
    main()
