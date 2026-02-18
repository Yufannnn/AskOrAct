"""Run full evaluation sweep; save results and plots."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import run_sweep, plot_regret_vs_ambiguity, plot_questions_vs_ambiguity

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    csv_path = os.path.join(results_dir, "metrics.csv")
    json_path = os.path.join(results_dir, "summary.json")
    rows, summary = run_sweep(output_csv=csv_path, output_json=json_path)
    print("Wrote", csv_path, "(%d rows)" % len(rows))
    print("Wrote", json_path)
    plot_regret_vs_ambiguity(csv_path=csv_path, output_path=os.path.join(results_dir, "regret_vs_ambiguity.png"))
    plot_questions_vs_ambiguity(csv_path=csv_path, output_path=os.path.join(results_dir, "questions_vs_ambiguity.png"))

if __name__ == "__main__":
    main()
