"""Run focused robustness sweeps for K={3,4} and save delta plots."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import (
    run_robust_answer_noise,
    run_robust_mismatch,
    plot_robust_answer_noise_deltas,
    plot_robust_mismatch_deltas,
)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    csv_a = os.path.join(results_dir, "metrics_robust_answer_noise.csv")
    rows_a = run_robust_answer_noise(output_csv=csv_a)
    print("Wrote", csv_a, "(%d rows)" % len(rows_a))
    plot_robust_answer_noise_deltas(
        csv_path=csv_a,
        output_path=os.path.join(results_dir, "robust_answer_noise_deltas.png"),
    )

    csv_b = os.path.join(results_dir, "metrics_robust_mismatch.csv")
    rows_b = run_robust_mismatch(output_csv=csv_b)
    print("Wrote", csv_b, "(%d rows)" % len(rows_b))
    plot_robust_mismatch_deltas(
        csv_path=csv_b,
        output_path=os.path.join(results_dir, "robust_mismatch_deltas.png"),
    )


if __name__ == "__main__":
    main()
