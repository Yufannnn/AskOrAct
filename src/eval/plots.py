"""Read metrics CSV and produce evaluation plots."""

import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _parse_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def bootstrap_mean_ci(values, n_boot=1000, rng_seed=0):
    """Return (mean, ci_lo, ci_hi) for the sample mean using bootstrap percentiles."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    sample_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(sample_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def load_metrics(csv_path="results/metrics.csv"):
    if not os.path.isfile(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["ambiguity_K"] = int(row["ambiguity_K"])
            row["eps"] = float(row["eps"])
            row["beta"] = float(row["beta"])
            row["success"] = _parse_bool(row["success"])
            row["steps"] = int(row["steps"])
            row["questions_asked"] = int(row["questions_asked"])
            row["regret"] = float(row["regret"])
            if "assistant_picked_goal" in row:
                row["assistant_picked_goal"] = _parse_bool(row["assistant_picked_goal"])
            if "final_map_correct" in row:
                row["final_map_correct"] = _parse_bool(row["final_map_correct"])
            if "oracle_steps" in row:
                row["oracle_steps"] = float(row["oracle_steps"]) if row["oracle_steps"] else 0
            if "episode_max_steps" in row:
                row["episode_max_steps"] = int(row["episode_max_steps"]) if row["episode_max_steps"] else 0
            if "team_cost" in row:
                row["team_cost"] = float(row["team_cost"]) if row["team_cost"] else 0.0
            if "oracle_cost" in row:
                row["oracle_cost"] = float(row["oracle_cost"]) if row["oracle_cost"] else 0.0
            if "terminated_by_wrong_pick" in row:
                row["terminated_by_wrong_pick"] = _parse_bool(row["terminated_by_wrong_pick"])
            if "failure_by_wrong_pick" in row:
                row["failure_by_wrong_pick"] = _parse_bool(row["failure_by_wrong_pick"])
            if "failure_by_timeout" in row:
                row["failure_by_timeout"] = _parse_bool(row["failure_by_timeout"])
            if "rep_seed" in row:
                row["rep_seed"] = int(row["rep_seed"])
            if "episode_id" in row:
                row["episode_id"] = int(row["episode_id"])
            if "entropy_before_first_ask" in row:
                row["entropy_before_first_ask"] = float(row["entropy_before_first_ask"])
            if "entropy_after_first_ask" in row:
                row["entropy_after_first_ask"] = float(row["entropy_after_first_ask"])
            if "effective_goal_count_before" in row:
                row["effective_goal_count_before"] = float(row["effective_goal_count_before"])
            if "effective_goal_count_after" in row:
                row["effective_goal_count_after"] = float(row["effective_goal_count_after"])
            if "ig_of_first_asked_question" in row:
                row["ig_of_first_asked_question"] = float(row["ig_of_first_asked_question"])
            rows.append(row)
    return rows


def load_ablation_metrics(csv_path="results/metrics_ablations.csv"):
    if not os.path.isfile(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["K"] = int(row["K"])
            row["eps"] = float(row["eps"])
            row["answer_noise"] = float(row["answer_noise"])
            row["question_cost"] = float(row["question_cost"])
            row["ask_counts_as_step"] = _parse_bool(row["ask_counts_as_step"])
            row["wrong_pick_fail"] = _parse_bool(row["wrong_pick_fail"])
            row["deadline_margin"] = int(row["deadline_margin"])
            row["success"] = _parse_bool(row["success"])
            row["steps"] = int(row["steps"])
            row["questions_asked"] = int(row["questions_asked"])
            row["regret"] = float(row["regret"])
            row["map_correct"] = _parse_bool(row["map_correct"])
            row["episode_max_steps"] = int(row["episode_max_steps"])
            if "failure_by_wrong_pick" in row:
                row["failure_by_wrong_pick"] = _parse_bool(row["failure_by_wrong_pick"])
            if "failure_by_timeout" in row:
                row["failure_by_timeout"] = _parse_bool(row["failure_by_timeout"])
            if "rep_seed" in row:
                row["rep_seed"] = int(row["rep_seed"])
            if "episode_id" in row:
                row["episode_id"] = int(row["episode_id"])
            if "entropy_before_first_ask" in row:
                row["entropy_before_first_ask"] = float(row["entropy_before_first_ask"])
            if "entropy_after_first_ask" in row:
                row["entropy_after_first_ask"] = float(row["entropy_after_first_ask"])
            if "effective_goal_count_before" in row:
                row["effective_goal_count_before"] = float(row["effective_goal_count_before"])
            if "effective_goal_count_after" in row:
                row["effective_goal_count_after"] = float(row["effective_goal_count_after"])
            if "ig_of_first_asked_question" in row:
                row["ig_of_first_asked_question"] = float(row["ig_of_first_asked_question"])
            rows.append(row)
    return rows


def _load_robust_metrics(csv_path):
    if not os.path.isfile(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for key in (
                "K",
                "rep_seed",
                "episode_id",
                "steps",
                "questions_asked",
                "episode_max_steps",
            ):
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            for key in (
                "answer_noise",
                "principal_eps",
                "assistant_eps",
                "principal_beta",
                "assistant_beta",
                "regret",
                "entropy_before_first_ask",
                "entropy_after_first_ask",
                "effective_goal_count_before",
                "effective_goal_count_after",
                "ig_of_first_asked_question",
            ):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            for key in ("success", "map_correct", "failure_by_wrong_pick", "failure_by_timeout"):
                if key in row:
                    row[key] = _parse_bool(row[key])
            rows.append(row)
    return rows


def _paired_delta(values_a, values_b, n_boot=1000, rng_seed=0):
    """Bootstrap CI for mean(values_a - values_b), paired by index order."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return np.nan, np.nan, np.nan
    d = a[:n] - b[:n]
    return bootstrap_mean_ci(d, n_boot=n_boot, rng_seed=rng_seed)


def _paired_delta_by_key(rows, cond_key_fn, metric_fn, policy_a, policy_b):
    """Return dict cond_key -> (mean, lo, hi, n) for paired delta (policy_a - policy_b)."""
    out = {}
    cond_keys = sorted(set(cond_key_fn(r) for r in rows))
    for ck in cond_keys:
        a_rows = [r for r in rows if cond_key_fn(r) == ck and r["policy"] == policy_a]
        b_rows = [r for r in rows if cond_key_fn(r) == ck and r["policy"] == policy_b]
        map_a = {
            (r.get("rep_seed", 0), r.get("episode_id", 0)): metric_fn(r)
            for r in a_rows
        }
        map_b = {
            (r.get("rep_seed", 0), r.get("episode_id", 0)): metric_fn(r)
            for r in b_rows
        }
        common = sorted(set(map_a.keys()) & set(map_b.keys()))
        if not common:
            out[ck] = (np.nan, np.nan, np.nan, 0)
            continue
        vals_a = [map_a[k] for k in common]
        vals_b = [map_b[k] for k in common]
        mean, lo, hi = _paired_delta(vals_a, vals_b, n_boot=1000, rng_seed=9000 + len(common))
        out[ck] = (mean, lo, hi, len(common))
    return out


def plot_robust_answer_noise_deltas(
    csv_path="results/metrics_robust_answer_noise.csv",
    output_path="results/robust_answer_noise_deltas.png",
):
    """Plot DeltaSuccess and DeltaRegret vs answer_noise, split by K in {3,4}."""
    if plt is None:
        return
    rows = _load_robust_metrics(csv_path)
    if not rows:
        print("No robustness metrics found at", csv_path)
        return
    ks = sorted(set(int(r["K"]) for r in rows))
    x_vals = sorted(set(float(r["answer_noise"]) for r in rows))

    def cond_key_fn(r):
        return (int(r["K"]), float(r["answer_noise"]))

    dsucc = _paired_delta_by_key(
        rows,
        cond_key_fn=cond_key_fn,
        metric_fn=lambda r: 1.0 if r["success"] else 0.0,
        policy_a="ask_or_act",
        policy_b="never_ask",
    )
    dreg = _paired_delta_by_key(
        rows,
        cond_key_fn=cond_key_fn,
        metric_fn=lambda r: float(r["regret"]),
        policy_a="ask_or_act",
        policy_b="always_ask",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for panel_idx, (ax, data, title, ylabel) in enumerate(
        [
            (axes[0], dsucc, "DeltaSuccess vs answer_noise", "DeltaSuccess (AskOrAct - NeverAsk)"),
            (axes[1], dreg, "DeltaRegret vs answer_noise", "DeltaRegret (AskOrAct - AlwaysAsk)"),
        ]
    ):
        for k_idx, K in enumerate(ks):
            means, los, his = [], [], []
            for x in x_vals:
                m, lo, hi, _n = data.get((K, x), (np.nan, np.nan, np.nan, 0))
                means.append(m)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(x_vals, means, marker="o", label=f"K={K}")
            ax.fill_between(x_vals, los, his, alpha=0.2)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("answer_noise")
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (95% bootstrap CI)")
        ax.legend()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_robust_mismatch_deltas(
    csv_path="results/metrics_robust_mismatch.csv",
    output_path="results/robust_mismatch_deltas.png",
):
    """Plot DeltaSuccess and DeltaRegret vs principal_beta mismatch, split by K in {3,4}."""
    if plt is None:
        return
    rows = _load_robust_metrics(csv_path)
    if not rows:
        print("No robustness metrics found at", csv_path)
        return
    ks = sorted(set(int(r["K"]) for r in rows))
    x_vals = sorted(set(float(r["principal_beta"]) for r in rows))

    def cond_key_fn(r):
        return (int(r["K"]), float(r["principal_beta"]))

    dsucc = _paired_delta_by_key(
        rows,
        cond_key_fn=cond_key_fn,
        metric_fn=lambda r: 1.0 if r["success"] else 0.0,
        policy_a="ask_or_act",
        policy_b="never_ask",
    )
    dreg = _paired_delta_by_key(
        rows,
        cond_key_fn=cond_key_fn,
        metric_fn=lambda r: float(r["regret"]),
        policy_a="ask_or_act",
        policy_b="always_ask",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for panel_idx, (ax, data, title, ylabel) in enumerate(
        [
            (axes[0], dsucc, "DeltaSuccess vs principal_beta", "DeltaSuccess (AskOrAct - NeverAsk)"),
            (axes[1], dreg, "DeltaRegret vs principal_beta", "DeltaRegret (AskOrAct - AlwaysAsk)"),
        ]
    ):
        for k_idx, K in enumerate(ks):
            means, los, his = [], [], []
            for x in x_vals:
                m, lo, hi, _n = data.get((K, x), (np.nan, np.nan, np.nan, 0))
                means.append(m)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(x_vals, means, marker="o", label=f"K={K}")
            ax.fill_between(x_vals, los, his, alpha=0.2)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("principal_beta")
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (95% bootstrap CI)")
        ax.legend()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def aggregate_by_condition(rows):
    from collections import defaultdict

    key_to_vals = defaultdict(lambda: {"success": [], "steps": [], "questions": [], "regret": [], "map_correct": []})
    for r in rows:
        key = (r["ambiguity_K"], r["eps"], r["beta"], r["policy"])
        key_to_vals[key]["success"].append(1 if r["success"] else 0)
        key_to_vals[key]["steps"].append(r["steps"])
        key_to_vals[key]["questions"].append(r["questions_asked"])
        key_to_vals[key]["regret"].append(r["regret"])
        if "final_map_correct" in r:
            key_to_vals[key]["map_correct"].append(1 if r["final_map_correct"] else 0)
    out = {}
    for k, v in key_to_vals.items():
        out[k] = {
            "success_rate": np.mean(v["success"]),
            "avg_steps": np.mean(v["steps"]),
            "avg_questions": np.mean(v["questions"]),
            "avg_regret": np.mean(v["regret"]),
            "map_correct_rate": np.mean(v["map_correct"]) if v["map_correct"] else np.nan,
        }
    return out


def _plot_metric_vs_k(rows, metric_fn, ylabel, title, output_path, ylim=None):
    if plt is None:
        return
    if not rows:
        return
    k_vals = sorted(set(r["K"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    plt.figure()
    for policy in policies:
        ys = []
        for K in k_vals:
            sub = [r for r in rows if r["K"] == K and r["policy"] == policy]
            ys.append(metric_fn(sub) if sub else np.nan)
        plt.plot(k_vals, ys, marker="o", label=policy)
    plt.xlabel("Ambiguity K")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.title(title)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print("Saved", output_path)


def plot_regret_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/regret_vs_ambiguity.png"):
    rows = load_metrics(csv_path)
    if not rows:
        print("No metrics found at", csv_path)
        return
    agg = aggregate_by_condition(rows)
    collapsed = []
    for (K, _eps, _beta, policy), s in agg.items():
        collapsed.append({"K": K, "policy": policy, "value": s["avg_regret"]})
    _plot_metric_vs_k(
        collapsed,
        metric_fn=lambda sub: float(np.mean([x["value"] for x in sub])),
        ylabel="Average regret",
        title="Regret vs K",
        output_path=output_path,
    )


def plot_questions_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/questions_vs_ambiguity.png"):
    rows = load_metrics(csv_path)
    if not rows:
        return
    agg = aggregate_by_condition(rows)
    collapsed = []
    for (K, _eps, _beta, policy), s in agg.items():
        collapsed.append({"K": K, "policy": policy, "value": s["avg_questions"]})
    _plot_metric_vs_k(
        collapsed,
        metric_fn=lambda sub: float(np.mean([x["value"] for x in sub])),
        ylabel="Average questions asked",
        title="Questions vs K",
        output_path=output_path,
    )


def plot_success_rate_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/success_vs_K.png"):
    rows = load_metrics(csv_path)
    if not rows:
        return
    agg = aggregate_by_condition(rows)
    collapsed = []
    for (K, _eps, _beta, policy), s in agg.items():
        collapsed.append({"K": K, "policy": policy, "value": s["success_rate"]})
    _plot_metric_vs_k(
        collapsed,
        metric_fn=lambda sub: float(np.mean([x["value"] for x in sub])),
        ylabel="Success rate",
        title="Success rate vs K",
        output_path=output_path,
        ylim=(0, 1),
    )


def plot_map_rate_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/map_vs_K.png"):
    rows = load_metrics(csv_path)
    if not rows:
        return
    agg = aggregate_by_condition(rows)
    collapsed = []
    for (K, _eps, _beta, policy), s in agg.items():
        collapsed.append({"K": K, "policy": policy, "value": s.get("map_correct_rate", np.nan)})
    _plot_metric_vs_k(
        collapsed,
        metric_fn=lambda sub: float(np.nanmean([x["value"] for x in sub])),
        ylabel="MAP accuracy",
        title="MAP accuracy vs K",
        output_path=output_path,
        ylim=(0, 1),
    )


def _plot_ablation_metric(rows, metric_key, ylabel, title, output_path, ylim=None):
    _plot_metric_vs_k(
        rows,
        metric_fn=lambda sub: float(np.mean([x[metric_key] for x in sub])),
        ylabel=ylabel,
        title=title,
        output_path=output_path,
        ylim=ylim,
    )


def plot_ablation_figures(csv_path="results/metrics_ablations.csv", output_dir="results"):
    rows = load_ablation_metrics(csv_path)
    if not rows:
        print("No ablation metrics found at", csv_path)
        return

    modes = sorted(set(r["mode_name"] for r in rows))
    wrong_flags = sorted(set(r["wrong_pick_fail"] for r in rows))

    for mode in modes:
        for wrong_pick_fail in wrong_flags:
            sub = [r for r in rows if r["mode_name"] == mode and r["wrong_pick_fail"] == wrong_pick_fail]
            if not sub:
                continue
            suffix = f"{mode}__wrongpick_{1 if wrong_pick_fail else 0}"
            _plot_ablation_metric(
                sub,
                metric_key="success",
                ylabel="Success rate",
                title=f"Success vs K ({mode}, wrong_pick_fail={wrong_pick_fail})",
                output_path=os.path.join(output_dir, f"success_vs_K__{suffix}.png"),
                ylim=(0, 1),
            )
            _plot_ablation_metric(
                sub,
                metric_key="regret",
                ylabel="Average regret",
                title=f"Regret vs K ({mode}, wrong_pick_fail={wrong_pick_fail})",
                output_path=os.path.join(output_dir, f"regret_vs_K__{suffix}.png"),
            )
            _plot_ablation_metric(
                sub,
                metric_key="map_correct",
                ylabel="MAP accuracy",
                title=f"MAP vs K ({mode}, wrong_pick_fail={wrong_pick_fail})",
                output_path=os.path.join(output_dir, f"map_vs_K__{suffix}.png"),
                ylim=(0, 1),
            )

        # Convenience aliases for wrong_pick_fail=False
        base_sub = [r for r in rows if r["mode_name"] == mode and not r["wrong_pick_fail"]]
        if base_sub:
            _plot_ablation_metric(
                base_sub,
                metric_key="success",
                ylabel="Success rate",
                title=f"Success vs K ({mode})",
                output_path=os.path.join(output_dir, f"success_vs_K__{mode}.png"),
                ylim=(0, 1),
            )
            _plot_ablation_metric(
                base_sub,
                metric_key="regret",
                ylabel="Average regret",
                title=f"Regret vs K ({mode})",
                output_path=os.path.join(output_dir, f"regret_vs_K__{mode}.png"),
            )
            _plot_ablation_metric(
                base_sub,
                metric_key="map_correct",
                ylabel="MAP accuracy",
                title=f"MAP vs K ({mode})",
                output_path=os.path.join(output_dir, f"map_vs_K__{mode}.png"),
                ylim=(0, 1),
            )


def _main_collapsed(rows):
    agg = aggregate_by_condition(rows)
    out = []
    for (k, _eps, _beta, policy), s in agg.items():
        out.append({
            "K": k,
            "policy": policy,
            "success_rate": s["success_rate"],
            "avg_regret": s["avg_regret"],
            "avg_questions": s["avg_questions"],
            "map_correct_rate": s.get("map_correct_rate", np.nan),
        })
    return out


def plot_main_dashboard(csv_path="results/metrics.csv", output_path="results/main_dashboard.png"):
    """Single 2x2 dashboard for main sweep with 95% bootstrap CIs."""
    if plt is None:
        return
    rows = load_metrics(csv_path)
    if not rows:
        print("No metrics found at", csv_path)
        return
    k_vals = sorted(set(r["ambiguity_K"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        ("success", "Success Rate", (0, 1)),
        ("regret", "Average Regret", None),
        ("questions_asked", "Average Questions", None),
        ("final_map_correct", "MAP Accuracy", (0, 1)),
    ]
    metric_cast = {
        "success": lambda r: 1.0 if r["success"] else 0.0,
        "regret": lambda r: float(r["regret"]),
        "questions_asked": lambda r: float(r["questions_asked"]),
        "final_map_correct": lambda r: 1.0 if r.get("final_map_correct", False) else 0.0,
    }

    for panel_idx, (ax, (key, title, ylim)) in enumerate(zip(axes.flat, panels)):
        for pol_idx, policy in enumerate(policies):
            means, los, his = [], [], []
            for K in k_vals:
                vals = [metric_cast[key](r) for r in rows if r["policy"] == policy and r["ambiguity_K"] == K]
                mean, lo, hi = bootstrap_mean_ci(vals, n_boot=1000, rng_seed=1000 + panel_idx * 100 + pol_idx * 10 + K)
                means.append(mean)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(k_vals, means, marker="o", label=policy)
            ax.fill_between(k_vals, los, his, alpha=0.2)
        ax.set_xlabel("Ambiguity K")
        ax.set_ylabel(title)
        ax.set_title(title + " vs K (95% bootstrap CI)")
        if ylim is not None:
            ax.set_ylim(*ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_ablations_dashboard(csv_path="results/metrics_ablations.csv", output_path="results/ablations_dashboard.png"):
    """
    Single 2x3 dashboard for ablations:
    rows = wrong_pick_fail {False, True}
    cols = success, regret, MAP
    lines = policy + mode combination.
    """
    if plt is None:
        return
    rows = load_ablation_metrics(csv_path)
    if not rows:
        print("No ablation metrics found at", csv_path)
        return

    k_vals = sorted(set(r["K"] for r in rows))
    mode_vals = sorted(set(r["mode_name"] for r in rows))
    policy_vals = sorted(set(r["policy"] for r in rows))
    wrong_vals = [False, True]

    # Aggregate across eps + deadline_margin for readability.
    from collections import defaultdict
    agg = defaultdict(lambda: {"success": [], "regret": [], "map_correct": []})
    for r in rows:
        key = (r["wrong_pick_fail"], r["mode_name"], r["policy"], r["K"])
        agg[key]["success"].append(1 if r["success"] else 0)
        agg[key]["regret"].append(r["regret"])
        agg[key]["map_correct"].append(1 if r["map_correct"] else 0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metric_defs = [
        ("success", "Success Rate", (0, 1)),
        ("regret", "Average Regret", None),
        ("map_correct", "MAP Accuracy", (0, 1)),
    ]

    for i, wrong_pick_fail in enumerate(wrong_vals):
        for j, (metric_key, metric_label, ylim) in enumerate(metric_defs):
            ax = axes[i, j]
            for mode in mode_vals:
                for policy in policy_vals:
                    ys = []
                    for K in k_vals:
                        vals = agg[(wrong_pick_fail, mode, policy, K)][metric_key]
                        ys.append(float(np.mean(vals)) if vals else np.nan)
                    ax.plot(k_vals, ys, marker="o", linewidth=1.5, label=f"{mode}:{policy}")
            ax.set_xlabel("Ambiguity K")
            ax.set_ylabel(metric_label)
            row_label = "wrong_pick_fail=1" if wrong_pick_fail else "wrong_pick_fail=0"
            ax.set_title(f"{metric_label} ({row_label})")
            if ylim is not None:
                ax.set_ylim(*ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_clarification_quality_entropy_delta(
    csv_path="results/metrics.csv",
    output_path="results/clarification_quality_entropy_delta.png",
):
    """
    Plot delta entropy (before first ask - after first ask) for K in {3,4}, by policy.
    Policies that never ask have delta ~0 by construction.
    """
    if plt is None:
        return
    rows = load_metrics(csv_path)
    if not rows:
        print("No metrics found at", csv_path)
        return

    rows = [r for r in rows if r["ambiguity_K"] in (3, 4)]
    if not rows:
        print("No K=3/4 rows found at", csv_path)
        return

    policies = sorted(set(r["policy"] for r in rows))
    k_vals = [3, 4]

    x = np.arange(len(policies), dtype=float)
    width = 0.35
    offsets = {3: -width / 2.0, 4: width / 2.0}

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), constrained_layout=True)
    for k in k_vals:
        means, err_low, err_high = [], [], []
        for pol_idx, policy in enumerate(policies):
            vals = [
                float(r.get("entropy_before_first_ask", 0.0)) - float(r.get("entropy_after_first_ask", 0.0))
                for r in rows
                if r["ambiguity_K"] == k and r["policy"] == policy
            ]
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot=1000, rng_seed=6000 + 100 * k + pol_idx)
            means.append(mean)
            err_low.append(max(0.0, mean - lo))
            err_high.append(max(0.0, hi - mean))
        ax.bar(
            x + offsets[k],
            means,
            width=width,
            label=f"K={k}",
            yerr=np.vstack([err_low, err_high]),
            capsize=3,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.set_ylabel("Delta entropy (before first ask - after first ask)")
    ax.set_title("Clarification quality by policy (K=3,4; 95% bootstrap CI)")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
    ax.legend()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def load_question_difficulty_metrics(csv_path="results/metrics_question_difficulty.csv"):
    return _load_robust_metrics(csv_path)


def plot_question_difficulty_dashboard(
    csv_path="results/metrics_question_difficulty.csv",
    output_path="results/question_difficulty_dashboard.png",
):
    """
    Focused 2x2 dashboard for question-difficulty experiment on K={3,4}.
    Panels: success, regret, questions, MAP with 95% bootstrap CI.
    """
    if plt is None:
        return
    rows = load_question_difficulty_metrics(csv_path)
    if not rows:
        print("No question-difficulty metrics found at", csv_path)
        return

    k_vals = sorted(set(int(r["K"]) for r in rows))
    policies = sorted(set(r["policy"] for r in rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    panels = [
        ("success", "Success Rate", (0, 1), lambda r: 1.0 if r["success"] else 0.0),
        ("regret", "Average Regret", None, lambda r: float(r["regret"])),
        ("questions_asked", "Average Questions", None, lambda r: float(r["questions_asked"])),
        ("map_correct", "MAP Accuracy", (0, 1), lambda r: 1.0 if r.get("map_correct", False) else 0.0),
    ]

    for panel_idx, (ax, (metric_key, title, ylim, metric_fn)) in enumerate(zip(axes.flat, panels)):
        for pol_idx, policy in enumerate(policies):
            means, los, his = [], [], []
            for K in k_vals:
                vals = [metric_fn(r) for r in rows if int(r["K"]) == K and r["policy"] == policy]
                mean, lo, hi = bootstrap_mean_ci(
                    vals,
                    n_boot=1000,
                    rng_seed=13000 + panel_idx * 100 + pol_idx * 10 + int(K),
                )
                means.append(mean)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(k_vals, means, marker="o", label=policy)
            ax.fill_between(k_vals, los, his, alpha=0.2)
        ax.set_xlabel("Ambiguity K")
        ax.set_ylabel(title)
        ax.set_title(title + " vs K (95% bootstrap CI)")
        if ylim is not None:
            ax.set_ylim(*ylim)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_question_difficulty_entropy_delta(
    csv_path="results/metrics_question_difficulty.csv",
    output_path="results/question_difficulty_entropy_delta.png",
):
    """
    Clarification quality in question-difficulty experiment:
    delta entropy = H(before first ask) - H(after first ask), by policy and K.
    """
    if plt is None:
        return
    rows = load_question_difficulty_metrics(csv_path)
    if not rows:
        print("No question-difficulty metrics found at", csv_path)
        return

    k_vals = sorted(set(int(r["K"]) for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    x = np.arange(len(policies), dtype=float)
    width = 0.35 if len(k_vals) <= 2 else 0.24
    offsets = np.linspace(-width * (len(k_vals) - 1) / 2, width * (len(k_vals) - 1) / 2, len(k_vals))

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), constrained_layout=True)
    for idx, K in enumerate(k_vals):
        means, err_low, err_high = [], [], []
        for pol_idx, policy in enumerate(policies):
            vals = [
                float(r.get("entropy_before_first_ask", 0.0)) - float(r.get("entropy_after_first_ask", 0.0))
                for r in rows
                if int(r["K"]) == K and r["policy"] == policy
            ]
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot=1000, rng_seed=14000 + idx * 100 + pol_idx)
            means.append(mean)
            err_low.append(max(0.0, mean - lo))
            err_high.append(max(0.0, hi - mean))
        ax.bar(
            x + offsets[idx],
            means,
            width=width,
            label=f"K={K}",
            yerr=np.vstack([err_low, err_high]),
            capsize=3,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.set_ylabel("Delta entropy (before first ask - after first ask)")
    ax.set_title("Question-difficulty clarification quality (95% bootstrap CI)")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
    ax.legend()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def load_generalization_templates_metrics(csv_path="results/metrics_generalization_templates.csv"):
    return _load_robust_metrics(csv_path)


def load_scale_k_metrics(csv_path="results/metrics_scaleK.csv"):
    return _load_robust_metrics(csv_path)


def plot_generalization_templates(
    csv_path="results/metrics_generalization_templates.csv",
    output_path="results/generalization_templates_plot.png",
):
    """
    Held-out template evaluation plot with CIs:
    success/regret/questions vs K, line per policy.
    """
    if plt is None:
        return
    rows = load_generalization_templates_metrics(csv_path)
    if not rows:
        print("No generalization-template metrics found at", csv_path)
        return

    k_vals = sorted(set(int(r["K"]) for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    panels = [
        ("success", "Success Rate", (0, 1), lambda r: 1.0 if r["success"] else 0.0),
        ("regret", "Average Regret", None, lambda r: float(r["regret"])),
        ("questions_asked", "Average Questions", None, lambda r: float(r["questions_asked"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for panel_idx, (ax, (metric_name, ylabel, ylim, metric_fn)) in enumerate(zip(axes, panels)):
        for pol_idx, policy in enumerate(policies):
            means, los, his = [], [], []
            for K in k_vals:
                vals = [metric_fn(r) for r in rows if int(r["K"]) == K and r["policy"] == policy]
                mean, lo, hi = bootstrap_mean_ci(
                    vals, n_boot=1000, rng_seed=15000 + panel_idx * 100 + pol_idx * 10 + int(K)
                )
                means.append(mean)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(k_vals, means, marker="o", label=policy)
            ax.fill_between(k_vals, los, his, alpha=0.2)
        ax.set_xlabel("Ambiguity K")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs K (held-out templates)")
        if ylim is not None:
            ax.set_ylim(*ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_scale_k(
    csv_path="results/metrics_scaleK.csv",
    output_path="results/scaleK_plot.png",
):
    """
    Scale-K summary plot:
      - regret vs K (up to 6)
      - questions vs K (up to 6)
    with 95% bootstrap CI.
    """
    if plt is None:
        return
    rows = load_scale_k_metrics(csv_path)
    if not rows:
        print("No scale-K metrics found at", csv_path)
        return

    k_vals = sorted(set(int(r["K"]) for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    panels = [
        ("regret", "Average Regret", lambda r: float(r["regret"])),
        ("questions_asked", "Average Questions", lambda r: float(r["questions_asked"])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for panel_idx, (ax, (metric_name, ylabel, metric_fn)) in enumerate(zip(axes, panels)):
        for pol_idx, policy in enumerate(policies):
            means, los, his = [], [], []
            for K in k_vals:
                vals = [metric_fn(r) for r in rows if int(r["K"]) == K and r["policy"] == policy]
                mean, lo, hi = bootstrap_mean_ci(
                    vals, n_boot=1000, rng_seed=16000 + panel_idx * 100 + pol_idx * 10 + int(K)
                )
                means.append(mean)
                los.append(lo)
                his.append(hi)
            means = np.asarray(means, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(k_vals, means, marker="o", label=policy)
            ax.fill_between(k_vals, los, his, alpha=0.2)
        ax.set_xlabel("Ambiguity K")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs K (eps=0.05, beta=2.0)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)


def plot_setup_overview(output_path="results/setup_overview.png"):
    """Draw a schematic of the cooperative task setup."""
    if plt is None:
        return
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.8), constrained_layout=True)
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    # Gridworld panel.
    grid_x0, grid_y0, cell = 0.5, 0.8, 0.9
    N = 9
    wall_color = "#3b3f4a"
    room_fill = "#f6f1e8"
    right_fill = "#eef3f8"
    doorway_row = 4
    split_col = 4

    for r in range(N):
        for c in range(N):
            x = grid_x0 + c * cell
            y = grid_y0 + (N - 1 - r) * cell
            fill = room_fill if c < split_col else right_fill
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=fill, edgecolor="#c8c8c8", linewidth=0.8))

    # Outer boundary.
    ax.add_patch(Rectangle((grid_x0, grid_y0), N * cell, N * cell, fill=False, edgecolor=wall_color, linewidth=2.0))

    # Internal wall with a doorway.
    wall_x = grid_x0 + split_col * cell
    for r in range(N):
        if r == doorway_row:
            continue
        y = grid_y0 + (N - 1 - r) * cell
        ax.add_patch(Rectangle((wall_x - 0.06, y), 0.12, cell, facecolor=wall_color, edgecolor=wall_color))
    door_y = grid_y0 + (N - 1 - doorway_row) * cell + 0.5 * cell
    ax.text(wall_x + 0.15, door_y, "door", fontsize=10, color=wall_color, va="center")

    def _cell_center(row, col):
        return grid_x0 + col * cell + 0.5 * cell, grid_y0 + (N - 1 - row) * cell + 0.5 * cell

    objects = [
        {"pos": (1, 1), "label": "blue gem", "face": "#4c78a8", "edge": "#1f3b5d"},
        {"pos": (1, 6), "label": "red key", "face": "#e45756", "edge": "#8a1c1c"},
        {"pos": (3, 2), "label": "red coin", "face": "#e45756", "edge": "#8a1c1c"},
        {"pos": (6, 7), "label": "red gem", "face": "#e45756", "edge": "#f2b701"},
        {"pos": (7, 2), "label": "green key", "face": "#54a24b", "edge": "#2c6626"},
    ]
    for obj in objects:
        cx, cy = _cell_center(*obj["pos"])
        ax.add_patch(Circle((cx, cy), 0.24, facecolor=obj["face"], edgecolor=obj["edge"], linewidth=2.2))

    # Agents.
    for row, col, label, color in [
        (2, 1, "P", "#222222"),
        (6, 1, "A", "#f58518"),
    ]:
        cx, cy = _cell_center(row, col)
        ax.add_patch(Circle((cx, cy), 0.28, facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    # Call out the true hidden goal.
    goal_x, goal_y = _cell_center(6, 7)
    ax.add_patch(Circle((goal_x, goal_y), 0.33, facecolor="none", edgecolor="#f2b701", linewidth=2.5, linestyle="--"))
    ax.annotate(
        "true goal (hidden from assistant)",
        xy=(goal_x + 0.05, goal_y + 0.1),
        xytext=(10.8, 8.2),
        arrowprops=dict(arrowstyle="->", color="#8a1c1c", linewidth=1.5),
        fontsize=10,
        color="#8a1c1c",
    )

    # Instruction and observations panel.
    ax.add_patch(
        FancyBboxPatch(
            (10.1, 6.9),
            6.8,
            2.2,
            boxstyle="round,pad=0.25,rounding_size=0.2",
            facecolor="#fff7e6",
            edgecolor="#c98f00",
            linewidth=1.4,
        )
    )
    ax.text(10.45, 8.55, "Instruction", fontsize=12, fontweight="bold", color="#7a4b00")
    ax.text(10.45, 7.85, "\"get red object\"", fontsize=14, color="#7a4b00")
    ax.text(10.45, 7.2, "K = 3 candidate goals share this surface form", fontsize=10, color="#7a4b00")

    ax.add_patch(
        FancyBboxPatch(
            (10.1, 4.0),
            6.8,
            1.9,
            boxstyle="round,pad=0.25,rounding_size=0.2",
            facecolor="#edf6ff",
            edgecolor="#4c78a8",
            linewidth=1.4,
        )
    )
    ax.text(10.45, 5.35, "Assistant observes", fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(10.45, 4.7, "- world state and instruction", fontsize=10, color="#1f4e79")
    ax.text(10.45, 4.2, "- principal actions over time", fontsize=10, color="#1f4e79")

    ax.add_patch(
        FancyBboxPatch(
            (10.1, 1.0),
            6.8,
            2.2,
            boxstyle="round,pad=0.25,rounding_size=0.2",
            facecolor="#eef8ee",
            edgecolor="#54a24b",
            linewidth=1.4,
        )
    )
    ax.text(10.45, 2.65, "Clarification menu", fontsize=12, fontweight="bold", color="#2f6b2f")
    ax.text(10.45, 2.0, "ask_color    ask_type    ask_room", fontsize=10, color="#2f6b2f")
    ax.text(10.45, 1.4, "answers update the posterior over candidate goals", fontsize=10, color="#2f6b2f")

    ax.add_patch(FancyArrowPatch((8.9, 7.4), (10.0, 7.9), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#7a4b00"))
    ax.add_patch(FancyArrowPatch((8.8, 4.8), (10.0, 4.9), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#1f4e79"))
    ax.add_patch(FancyArrowPatch((8.7, 2.0), (10.0, 2.0), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#2f6b2f"))

    ax.text(
        0.6,
        9.95,
        "Task setup: ambiguous instruction, hidden goal, and shared world state",
        fontsize=14,
        fontweight="bold",
        color="#202020",
    )
    ax.text(0.65, 9.35, "Example shown for K = 3: three red objects match the same instruction, but only one is the principal's goal.", fontsize=10.5, color="#404040")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print("Saved", output_path)


def plot_ask_or_act_pipeline(output_path="results/ask_or_act_pipeline.png"):
    """Draw a conceptual decision-flow diagram for AskOrAct."""
    if plt is None:
        return
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.4), constrained_layout=True)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def _box(x, y, w, h, title, body, face, edge):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.2,rounding_size=0.18",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
            )
        )
        ax.text(x + 0.2, y + h - 0.45, title, fontsize=11.5, fontweight="bold", color=edge)
        ax.text(x + 0.2, y + h - 0.95, body, fontsize=9.8, color="#303030", va="top")

    def _arrow(x0, y0, x1, y1, text=None, color="#5b5b5b", rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.5,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if text:
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.25, text, fontsize=9.5, color=color, ha="center")

    _box(0.6, 6.0, 2.3, 1.7, "1. Instruction u", "templated language\nidentifies K candidates", "#fff7e6", "#9a6700")
    _box(3.4, 6.0, 2.5, 1.7, "2. Candidate goals G", "objects consistent\nwith the instruction", "#f4f1fb", "#6c4aa4")
    _box(6.4, 6.0, 2.4, 1.7, "3. Posterior b_t(g)", "belief over goals\nstarts from instruction prior", "#edf6ff", "#356c9b")
    _box(3.4, 3.5, 2.5, 1.55, "4. Principal action a_t", "observe next move\nfrom the hidden-goal principal", "#fff3f0", "#b04a3f")
    _box(6.4, 3.35, 3.0, 1.9, "5. Bayesian update", "b_{t+1}(g) ∝ b_t(g) pi(a_t | s_t, g)\nplus answer-likelihood updates after a question", "#eef8ee", "#3d7d3d")
    _box(10.0, 5.55, 3.6, 2.15, "6. Evaluate decision", "CostAct = expected task cost now\nCostAsk = 1 + c_q + E[CostAct after answer]", "#fff7e6", "#9a6700")
    _box(10.1, 3.0, 3.4, 1.65, "Gates", "entropy gate\nask window\nquestion budget", "#f8f8f8", "#666666")
    _box(9.3, 0.8, 2.25, 1.4, "ASK", "choose best question\nif CostAsk < CostAct", "#eef8ee", "#3d7d3d")
    _box(11.95, 0.8, 2.25, 1.4, "ACT", "move toward MAP goal\nand pick when aligned", "#edf6ff", "#356c9b")

    _arrow(2.9, 6.85, 3.4, 6.85, color="#9a6700")
    _arrow(5.9, 6.85, 6.4, 6.85, color="#6c4aa4")
    _arrow(7.6, 6.0, 7.9, 5.25, color="#356c9b")
    _arrow(5.9, 4.3, 6.4, 4.3, text="inverse planning", color="#b04a3f")
    _arrow(9.4, 4.3, 10.0, 5.95, color="#3d7d3d")
    _arrow(11.7, 3.0, 11.7, 2.2, color="#666666")
    _arrow(11.1, 2.95, 10.4, 2.2, text="if ask", color="#3d7d3d", rad=0.05)
    _arrow(12.5, 2.95, 13.1, 2.2, text="if act", color="#356c9b", rad=-0.05)
    _arrow(10.4, 0.8, 7.4, 3.25, text="answer updates posterior", color="#3d7d3d", rad=0.15)
    _arrow(13.1, 0.8, 6.9, 6.0, text="new state and action evidence", color="#356c9b", rad=-0.22)

    ax.text(0.6, 8.45, "AskOrAct decision flow", fontsize=14, fontweight="bold", color="#202020")
    ax.text(0.6, 7.95, "The assistant alternates between belief updates and a one-step expected-cost comparison between asking and acting.", fontsize=10.5, color="#404040")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print("Saved", output_path)


def plot_pareto_k4(
    csv_path="results/metrics.csv",
    output_path="results/pareto_K4.png",
):
    """
    Pareto-style view at K=4:
      x = average questions asked
      y = average regret
    with bootstrap CI bars for each policy point.
    """
    if plt is None:
        return
    rows = load_metrics(csv_path)
    if not rows:
        print("No metrics found at", csv_path)
        return
    rows = [r for r in rows if int(r["ambiguity_K"]) == 4]
    if not rows:
        print("No K=4 rows found at", csv_path)
        return

    policies = sorted(set(r["policy"] for r in rows))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    for idx, policy in enumerate(policies):
        sub = [r for r in rows if r["policy"] == policy]
        if not sub:
            continue
        q_vals = [float(r["questions_asked"]) for r in sub]
        r_vals = [float(r["regret"]) for r in sub]
        q_mean, q_lo, q_hi = bootstrap_mean_ci(q_vals, n_boot=1000, rng_seed=18000 + idx)
        r_mean, r_lo, r_hi = bootstrap_mean_ci(r_vals, n_boot=1000, rng_seed=18100 + idx)
        xerr = np.array([[max(0.0, q_mean - q_lo)], [max(0.0, q_hi - q_mean)]], dtype=float)
        yerr = np.array([[max(0.0, r_mean - r_lo)], [max(0.0, r_hi - r_mean)]], dtype=float)
        ax.errorbar(
            q_mean,
            r_mean,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label=policy,
        )
        ax.annotate(policy, (q_mean, r_mean), textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xlabel("Average Questions (K=4)")
    ax.set_ylabel("Average Regret (K=4)")
    ax.set_title("Pareto View at K=4 (95% bootstrap CI)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved", output_path)
