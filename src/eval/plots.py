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
