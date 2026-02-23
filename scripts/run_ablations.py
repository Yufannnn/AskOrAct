"""Run fixed ablations and save metrics + plots."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import run_ablations, plot_ablations_dashboard


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    csv_path = os.path.join(results_dir, "metrics_ablations.csv")
    rows = run_ablations(output_csv=csv_path)
    print("Wrote", csv_path, "(%d rows)" % len(rows))
    plot_ablations_dashboard(csv_path=csv_path, output_path=os.path.join(results_dir, "ablations_dashboard.png"))


if __name__ == "__main__":
    main()
