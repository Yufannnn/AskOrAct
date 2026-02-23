"""Run focused question-difficulty sweep (K={3,4}) and generate plots."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import (
    run_question_difficulty_sweep,
    plot_question_difficulty_dashboard,
    plot_question_difficulty_entropy_delta,
)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    csv_path = os.path.join(results_dir, "metrics_question_difficulty.csv")
    rows = run_question_difficulty_sweep(output_csv=csv_path)
    print("Wrote", csv_path, "(%d rows)" % len(rows))

    plot_question_difficulty_dashboard(
        csv_path=csv_path,
        output_path=os.path.join(results_dir, "question_difficulty_dashboard.png"),
    )
    plot_question_difficulty_entropy_delta(
        csv_path=csv_path,
        output_path=os.path.join(results_dir, "question_difficulty_entropy_delta.png"),
    )


if __name__ == "__main__":
    main()

