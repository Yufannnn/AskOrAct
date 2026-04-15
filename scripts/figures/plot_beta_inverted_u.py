"""Plot success vs principal rationality beta at K=3 for the informative
irrationality sweep. Produces results/figures/paper/beta_inverted_u.png."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eval.plots import load_metrics, bootstrap_mean_ci  # noqa: E402

CSV_PATH = "results/metrics/metrics_informative_irrationality.csv"
OUT_PATH = "results/figures/paper/beta_inverted_u.png"
TARGET_K = 3

POLICY_ORDER = ["ask_or_act", "never_ask", "always_ask"]
POLICY_LABEL = {
    "ask_or_act": "AskOrAct",
    "never_ask":  "NeverAsk",
    "always_ask": "AlwaysAsk",
}
POLICY_STYLE = {
    "ask_or_act": {"marker": "o", "color": "#1f77b4"},
    "never_ask":  {"marker": "s", "color": "#ff7f0e"},
    "always_ask": {"marker": "^", "color": "#2ca02c"},
}


def _load_rows(path):
    rows = []
    import csv
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["K"] = int(row["K"])
            row["beta"] = float(row["beta"])
            row["success"] = row["success"] in ("True", "1", "true")
            rows.append(row)
    return rows


def main():
    rows = _load_rows(CSV_PATH)
    rows = [r for r in rows if r["K"] == TARGET_K]
    betas = sorted({r["beta"] for r in rows})

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for pol_idx, policy in enumerate(POLICY_ORDER):
        means, los, his = [], [], []
        for b in betas:
            vals = [1.0 if r["success"] else 0.0
                    for r in rows if r["policy"] == policy and r["beta"] == b]
            m, lo, hi = bootstrap_mean_ci(vals, n_boot=1000,
                                          rng_seed=4200 + pol_idx * 100 + int(b * 10))
            means.append(m * 100.0)
            los.append(lo * 100.0)
            his.append(hi * 100.0)
        style = POLICY_STYLE[policy]
        x = list(range(len(betas)))
        ax.plot(x, means, marker=style["marker"], color=style["color"],
                linewidth=1.6, markersize=5, label=POLICY_LABEL[policy])
        ax.fill_between(x, los, his, color=style["color"], alpha=0.15)

    ax.set_xticks(list(range(len(betas))))
    ax.set_xticklabels([f"{b:g}" for b in betas], fontsize=8)
    ax.set_xlabel(r"Principal rationality $\beta$", fontsize=9)
    ax.set_ylabel("Success rate (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print("Saved", OUT_PATH)


if __name__ == "__main__":
    main()
