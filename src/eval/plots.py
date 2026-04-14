"""Read metrics CSV and produce evaluation plots."""

import csv
import os
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
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


def load_metrics(csv_path="results/metrics/metrics.csv"):
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


def load_ablation_metrics(csv_path="results/metrics/metrics_ablations.csv"):
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
    csv_path="results/metrics/metrics_robust_answer_noise.csv",
    output_path="results/figures/exploratory/robust_answer_noise_deltas.png",
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
    csv_path="results/metrics/metrics_robust_mismatch.csv",
    output_path="results/figures/exploratory/robust_mismatch_deltas.png",
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


def plot_regret_vs_ambiguity(csv_path="results/metrics/metrics.csv", output_path="results/figures/exploratory/regret_vs_ambiguity.png"):
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


def plot_questions_vs_ambiguity(csv_path="results/metrics/metrics.csv", output_path="results/figures/exploratory/questions_vs_ambiguity.png"):
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


def plot_success_rate_vs_ambiguity(csv_path="results/metrics/metrics.csv", output_path="results/figures/exploratory/success_vs_K.png"):
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


def plot_map_rate_vs_ambiguity(csv_path="results/metrics/metrics.csv", output_path="results/figures/exploratory/map_vs_K.png"):
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


def plot_ablation_figures(csv_path="results/metrics/metrics_ablations.csv", output_dir="results/figures/exploratory"):
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


def plot_main_dashboard(csv_path="results/metrics/metrics.csv", output_path="results/figures/paper/main_dashboard.png"):
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


def plot_ablations_dashboard(csv_path="results/metrics/metrics_ablations.csv", output_path="results/figures/exploratory/ablations_dashboard.png"):
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
    csv_path="results/metrics/metrics.csv",
    output_path="results/figures/exploratory/clarification_quality_entropy_delta.png",
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


def load_question_difficulty_metrics(csv_path="results/metrics/metrics_question_difficulty.csv"):
    return _load_robust_metrics(csv_path)


def plot_question_difficulty_dashboard(
    csv_path="results/metrics/metrics_question_difficulty.csv",
    output_path="results/figures/exploratory/question_difficulty_dashboard.png",
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
    csv_path="results/metrics/metrics_question_difficulty.csv",
    output_path="results/figures/exploratory/question_difficulty_entropy_delta.png",
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


def load_generalization_templates_metrics(csv_path="results/metrics/metrics_generalization_templates.csv"):
    return _load_robust_metrics(csv_path)


def load_scale_k_metrics(csv_path="results/metrics/metrics_scaleK.csv"):
    return _load_robust_metrics(csv_path)


def plot_generalization_templates(
    csv_path="results/metrics/metrics_generalization_templates.csv",
    output_path="results/figures/exploratory/generalization_templates_plot.png",
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
    csv_path="results/metrics/metrics_scaleK.csv",
    output_path="results/figures/exploratory/scaleK_plot.png",
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


def plot_setup_overview(output_path="results/figures/paper/setup_overview.png"):
    """Draw a schematic of the cooperative task setup."""
    if plt is None:
        return
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.8), constrained_layout=True)
    ax.set_xlim(-0.5, 16.8)
    ax.set_ylim(-0.5, 10.8)
    ax.axis("off")

    # Gridworld panel — cell=1.0 makes the grid larger and better balanced.
    grid_x0, grid_y0, cell = 0.5, 0.7, 1.0
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
    ax.text(wall_x + 0.18, door_y, "door", fontsize=10, color=wall_color, va="center")

    def _cell_center(row, col):
        return grid_x0 + col * cell + 0.5 * cell, grid_y0 + (N - 1 - row) * cell + 0.5 * cell

    objects = [
        {"pos": (1, 1), "label": "blue gem",  "face": "#4c78a8", "edge": "#1f3b5d"},
        {"pos": (1, 6), "label": "red key",   "face": "#e45756", "edge": "#8a1c1c"},
        {"pos": (3, 2), "label": "red coin",  "face": "#e45756", "edge": "#8a1c1c"},
        {"pos": (6, 7), "label": "red gem",   "face": "#e45756", "edge": "#f2b701"},
        {"pos": (7, 2), "label": "green key", "face": "#54a24b", "edge": "#2c6626"},
    ]
    for obj in objects:
        cx, cy = _cell_center(*obj["pos"])
        ax.add_patch(Circle((cx, cy), 0.28, facecolor=obj["face"], edgecolor=obj["edge"], linewidth=2.2))

    # Agents.
    for row, col, label, color in [
        (2, 1, "P", "#222222"),
        (6, 1, "A", "#f58518"),
    ]:
        cx, cy = _cell_center(row, col)
        ax.add_patch(Circle((cx, cy), 0.32, facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    # Call out the true hidden goal — short arrow within the grid, avoids panels.
    goal_x, goal_y = _cell_center(6, 7)
    ax.add_patch(Circle((goal_x, goal_y), 0.38, facecolor="none", edgecolor="#f2b701", linewidth=2.5, linestyle="--"))
    ax.annotate(
        "true goal\n(hidden from assistant)",
        xy=(goal_x, goal_y + 0.38),
        xytext=(goal_x - 1.2, goal_y + 2.0),
        arrowprops=dict(arrowstyle="->", color="#8a1c1c", linewidth=1.3,
                        connectionstyle="arc3,rad=0.2"),
        fontsize=9.0,
        color="#8a1c1c",
        ha="center",
    )

    # Right-side panels — moved to x=9.9, width=6.0 to close the gap.
    panel_x, panel_w = 9.9, 6.0

    ax.add_patch(FancyBboxPatch(
        (panel_x, 6.9), panel_w, 2.2,
        boxstyle="round,pad=0.22,rounding_size=0.2",
        facecolor="#fff7e6", edgecolor="#c98f00", linewidth=1.4,
    ))
    ax.text(panel_x + 0.32, 8.58, "Instruction", fontsize=12, fontweight="bold", color="#7a4b00")
    ax.text(panel_x + 0.32, 7.88, "\"get red object\"", fontsize=13.5, color="#7a4b00")
    ax.text(panel_x + 0.32, 7.22, "K = 3 candidate goals share this surface form", fontsize=9.5, color="#7a4b00")

    ax.add_patch(FancyBboxPatch(
        (panel_x, 4.1), panel_w, 1.9,
        boxstyle="round,pad=0.22,rounding_size=0.2",
        facecolor="#edf6ff", edgecolor="#4c78a8", linewidth=1.4,
    ))
    ax.text(panel_x + 0.32, 5.42, "Assistant observes", fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(panel_x + 0.32, 4.78, "- world state and instruction", fontsize=9.5, color="#1f4e79")
    ax.text(panel_x + 0.32, 4.26, "- principal actions over time", fontsize=9.5, color="#1f4e79")

    ax.add_patch(FancyBboxPatch(
        (panel_x, 1.1), panel_w, 2.1,
        boxstyle="round,pad=0.22,rounding_size=0.2",
        facecolor="#eef8ee", edgecolor="#54a24b", linewidth=1.4,
    ))
    ax.text(panel_x + 0.32, 2.65, "Clarification menu", fontsize=12, fontweight="bold", color="#2f6b2f")
    ax.text(panel_x + 0.32, 2.02, "ask_color    ask_type    ask_room", fontsize=9.5, color="#2f6b2f")
    ax.text(panel_x + 0.32, 1.45, "answers update the posterior over candidate goals", fontsize=9.5, color="#2f6b2f")

    # Arrows from grid right edge to panels.
    ax.add_patch(FancyArrowPatch((9.6, 7.5), (panel_x - 0.1, 7.9), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#7a4b00"))
    ax.add_patch(FancyArrowPatch((9.6, 4.8), (panel_x - 0.1, 4.9), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#1f4e79"))
    ax.add_patch(FancyArrowPatch((9.6, 2.1), (panel_x - 0.1, 2.1), arrowstyle="->", mutation_scale=16, linewidth=1.5, color="#2f6b2f"))

    # Title and subtitle.
    ax.text(0.6, 10.48, "Task setup: ambiguous instruction, hidden goal, and shared world state",
            fontsize=13.5, fontweight="bold", color="#202020")
    ax.text(0.65, 9.98, "Example shown for K = 3: three red objects match the same instruction, "
            "but only one is the principal's goal.",
            fontsize=9.2, color="#404040", va="top")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print("Saved", output_path)


def plot_ask_or_act_pipeline(output_path="results/figures/paper/ask_or_act_pipeline.png"):
    """Draw a polished vertical decision-flow diagram for AskOrAct."""
    if plt is None:
        return
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch

    # --- Cohesive palette ---
    C = {
        "bg": "#F8FAFД",
        "node": "#F4F7FB",
        "node_edge": "#B8C5D9",
        "text": "#1F2937",
        "muted": "#6B7280",
        "accent1": "#4F6D8F",   # pipeline spine accent
        "ask": "#2563EB",       # blue
        "ask_bg": "#EFF6FF",
        "act": "#0D9488",       # teal
        "act_bg": "#F0FDFA",
        "decision": "#7C3AED",  # purple for decision node
        "decision_bg": "#F5F3FF",
        "guard": "#E5EAF3",
        "guard_edge": "#9CA3AF",
    }
    # Fix typo in hex (Cyrillic Д -> D)
    C["bg"] = "#F8FAFD"

    shadow_fx = [
        pe.withSimplePatchShadow(
            offset=(1.5, -1.5), shadow_rgbFace=(0.15, 0.2, 0.3), alpha=0.10
        ),
        pe.Normal(),
    ]

    arrow_kw = dict(
        arrowstyle="-|>",
        lw=2.0,
        color=C["accent1"],
        shrinkA=8,
        shrinkB=8,
        mutation_scale=14,
        joinstyle="round",
        capstyle="round",
    )

    W = 3.4
    H = 5.5
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 11.0)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    bw = 5.7
    bx = 0.55
    cx = bx + bw / 2

    def _box(y, h, label, body, face, edge, lw=1.4, shadow=True):
        p = FancyBboxPatch(
            (bx, y), bw, h,
            boxstyle="round,pad=0.07,rounding_size=0.16",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2,
        )
        if shadow:
            p.set_path_effects(shadow_fx)
        ax.add_patch(p)
        ax.text(cx, y + h - 0.14, label,
                fontsize=8.5, fontweight="semibold", color=edge,
                ha="center", va="top", zorder=3)
        ax.text(cx, y + h - 0.42, body,
                fontsize=7, color=C["text"], ha="center", va="top",
                linespacing=1.18, zorder=3)

    def _varrow(y0, y1, x=None, color=None):
        x = x if x is not None else cx
        kw = {**arrow_kw, "color": color or C["accent1"]}
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=kw)

    # ---- Pipeline nodes (top → bottom, compact spacing) ----
    def _numbered_box(y, h, num, title, body, face, edge, lw=1.4):
        """Draw a box with a circled step number badge on the left."""
        _box(y, h, title, body, face, edge, lw=lw)
        badge_x = bx + 0.35
        badge_y = y + h - 0.26
        badge = plt.Circle((badge_x, badge_y), 0.20,
                            facecolor=edge, edgecolor="white",
                            linewidth=1.0, zorder=4)
        ax.add_patch(badge)
        ax.text(badge_x, badge_y, str(num), fontsize=7, fontweight="bold",
                color="white", ha="center", va="center", zorder=5)

    _numbered_box(9.85, 0.85, 1, "Instruction  $u$",
                  "templated language identifies $K$ candidates",
                  "#FFF8F0", "#B45309")

    _numbered_box(8.55, 0.85, 2, "Candidate goals  $G$",
                  "objects matching the instruction",
                  C["node"], C["node_edge"])

    _numbered_box(7.25, 0.85, 3, "Posterior  $b(g)$",
                  "belief over goals, init uniform $1/K$",
                  C["node"], C["node_edge"])

    _numbered_box(5.70, 1.10, 4, "Bayesian update",
                  "observe principal action (inverse planning)\n"
                  "+ answer likelihood if question asked",
                  C["node"], C["accent1"], lw=1.6)

    _numbered_box(4.05, 1.10, 5, "Evaluate decision",
                  "$\\mathrm{CostAct} = \\mathbb{E}[\\mathrm{dist}]$\n"
                  "$\\mathrm{CostAsk} = 1 + c_q "
                  "+ \\mathbb{E}[\\mathrm{CostAct} \\mid \\mathrm{ans}]$",
                  C["decision_bg"], C["decision"], lw=1.6)

    # ---- Spine arrows ----
    _varrow(9.85, 9.40, color="#B45309")
    _varrow(8.55, 8.10, color=C["node_edge"])
    _varrow(7.25, 6.80, color=C["node_edge"])
    _varrow(5.70, 5.15, color=C["accent1"])

    # ---- ASK / ACT split ----
    askw = 2.65
    ask_p = FancyBboxPatch(
        (bx, 2.30), askw, 1.20,
        boxstyle="round,pad=0.07,rounding_size=0.16",
        facecolor=C["ask_bg"], edgecolor=C["ask"], linewidth=1.6, zorder=2,
    )
    ask_p.set_path_effects(shadow_fx)
    ax.add_patch(ask_p)
    # Accent top stripe for ASK
    ax.plot([bx + 0.16, bx + askw - 0.16], [3.48, 3.48],
            color=C["ask"], lw=2.8, solid_capstyle="round", zorder=3)
    ax.text(bx + askw / 2, 3.36, "ASK", fontsize=9.5, fontweight="bold",
            color=C["ask"], ha="center", va="top", zorder=3)
    ax.text(bx + askw / 2, 3.02,
            "select $q^* = \\arg\\min$ CostAsk\nupdate posterior",
            fontsize=7, color=C["text"], ha="center", va="top",
            linespacing=1.18, zorder=3)

    actx = bx + askw + 0.4
    actw = bw - askw - 0.4
    act_p = FancyBboxPatch(
        (actx, 2.30), actw, 1.20,
        boxstyle="round,pad=0.07,rounding_size=0.16",
        facecolor=C["act_bg"], edgecolor=C["act"], linewidth=1.6, zorder=2,
    )
    act_p.set_path_effects(shadow_fx)
    ax.add_patch(act_p)
    # Accent top stripe for ACT
    ax.plot([actx + 0.16, actx + actw - 0.16], [3.48, 3.48],
            color=C["act"], lw=2.8, solid_capstyle="round", zorder=3)
    ax.text(actx + actw / 2, 3.36, "ACT", fontsize=9.5, fontweight="bold",
            color=C["act"], ha="center", va="top", zorder=3)
    ax.text(actx + actw / 2, 3.02,
            "A* to MAP goal\npick up object",
            fontsize=7, color=C["text"], ha="center", va="top",
            linespacing=1.18, zorder=3)

    # ---- Decision → ASK / ACT arrows ----
    ask_cx = bx + askw / 2
    act_cx = actx + actw / 2
    ax.annotate("", xy=(ask_cx, 3.50), xytext=(cx - 0.5, 4.05),
                arrowprops={**arrow_kw, "color": C["ask"],
                            "connectionstyle": "arc3,rad=0.10"})
    ax.annotate("", xy=(act_cx, 3.50), xytext=(cx + 0.5, 4.05),
                arrowprops={**arrow_kw, "color": C["act"],
                            "connectionstyle": "arc3,rad=-0.10"})

    # ---- Guards footer bar ----
    guard_p = FancyBboxPatch(
        (bx, 1.55), bw, 0.50,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        facecolor=C["guard"], edgecolor=C["guard_edge"],
        linewidth=1.0, zorder=2,
    )
    ax.add_patch(guard_p)
    ax.text(cx, 1.80, "Guards:   $H(b) > 0.3$  ·  step $\\leq 6$  ·  $Q \\leq 3$",
            fontsize=7, color=C["muted"], ha="center", va="center", zorder=3,
            fontweight="medium")

    # Guards → ASK / ACT
    _varrow(2.05, 2.30, x=ask_cx, color=C["guard_edge"])
    _varrow(2.05, 2.30, x=act_cx, color=C["guard_edge"])

    # ---- Feedback loop: ASK → Bayesian update ----
    ax.annotate(
        "", xy=(bx - 0.05, 6.30), xytext=(bx - 0.05, 2.90),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.0,
            color=C["ask"],
            shrinkA=5,
            shrinkB=5,
            mutation_scale=13,
            connectionstyle="arc3,rad=0.28",
            joinstyle="round",
            capstyle="round",
        ),
    )
    ax.text(-0.10, 4.60, "loop", fontsize=7, color=C["ask"], rotation=90,
            ha="center", va="center", fontstyle="italic", fontweight="medium")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor=C["bg"])
    plt.close(fig)
    print("Saved", output_path)


def plot_pareto_k4(
    csv_path="results/metrics/metrics.csv",
    output_path="results/figures/exploratory/pareto_K4.png",
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
