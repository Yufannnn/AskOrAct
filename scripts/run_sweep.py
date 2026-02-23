"""Run full evaluation sweep; save results and plots."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import (
    run_sweep,
    plot_main_dashboard,
    plot_clarification_quality_entropy_delta,
)

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    csv_path = os.path.join(results_dir, "metrics.csv")
    json_path = os.path.join(results_dir, "summary.json")
    rows, summary = run_sweep(output_csv=csv_path, output_json=json_path)
    print("Wrote", csv_path, "(%d rows)" % len(rows))
    print("Wrote", json_path)
    plot_main_dashboard(csv_path=csv_path, output_path=os.path.join(results_dir, "main_dashboard.png"))
    plot_clarification_quality_entropy_delta(
        csv_path=csv_path,
        output_path=os.path.join(results_dir, "clarification_quality_entropy_delta.png"),
    )

if __name__ == "__main__":
    main()
