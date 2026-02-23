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
            rows.append(row)
    return rows


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
