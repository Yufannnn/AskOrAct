"""Read metrics CSV and plot regret and questions vs ambiguity."""

import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
            row["success"] = row["success"] == "True"
            row["steps"] = int(row["steps"])
            row["questions_asked"] = int(row["questions_asked"])
            row["regret"] = float(row["regret"])
            if "assistant_picked_goal" in row:
                row["assistant_picked_goal"] = row["assistant_picked_goal"] == "True"
            if "final_map_correct" in row:
                row["final_map_correct"] = row["final_map_correct"] == "True"
            if "oracle_steps" in row:
                row["oracle_steps"] = float(row["oracle_steps"]) if row["oracle_steps"] else 0
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
        }
        if v["map_correct"]:
            out[k]["map_correct_rate"] = np.mean(v["map_correct"])
    return out


def plot_regret_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/regret_vs_ambiguity.png"):
    if plt is None:
        print("matplotlib not available, skipping plot")
        return
    rows = load_metrics(csv_path)
    if not rows:
        print("No metrics found at", csv_path)
        return
    agg = aggregate_by_condition(rows)
    K_vals = sorted(set(r["ambiguity_K"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    for policy in policies:
        regrets = [np.mean([agg[k]["avg_regret"] for k in agg if k[0] == K and k[3] == policy]) if any(k[0] == K and k[3] == policy for k in agg) else np.nan for K in K_vals]
        plt.plot(K_vals, regrets, marker="o", label=policy)
    plt.xlabel("Ambiguity K")
    plt.ylabel("Average regret")
    plt.legend()
    plt.title("Regret vs ambiguity")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print("Saved", output_path)


def plot_questions_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/questions_vs_ambiguity.png"):
    if plt is None:
        return
    rows = load_metrics(csv_path)
    if not rows:
        return
    agg = aggregate_by_condition(rows)
    K_vals = sorted(set(r["ambiguity_K"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    for policy in policies:
        qs = [np.mean([agg[k]["avg_questions"] for k in agg if k[0] == K and k[3] == policy]) if any(k[0] == K and k[3] == policy for k in agg) else np.nan for K in K_vals]
        plt.plot(K_vals, qs, marker="o", label=policy)
    plt.xlabel("Ambiguity K")
    plt.ylabel("Average questions asked")
    plt.legend()
    plt.title("Questions asked vs ambiguity")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print("Saved", output_path)


def plot_success_rate_vs_ambiguity(csv_path="results/metrics.csv", output_path="results/success_rate_vs_ambiguity.png"):
    """Plot success rate vs ambiguity K per policy."""
    if plt is None:
        return
    rows = load_metrics(csv_path)
    if not rows:
        return
    agg = aggregate_by_condition(rows)
    K_vals = sorted(set(r["ambiguity_K"] for r in rows))
    policies = sorted(set(r["policy"] for r in rows))
    for policy in policies:
        success_rates = [np.mean([agg[k]["success_rate"] for k in agg if k[0] == K and k[3] == policy]) if any(k[0] == K and k[3] == policy for k in agg) else np.nan for K in K_vals]
        plt.plot(K_vals, success_rates, marker="o", label=policy)
    plt.xlabel("Ambiguity K")
    plt.ylabel("Success rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Success rate vs ambiguity")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print("Saved", output_path)
