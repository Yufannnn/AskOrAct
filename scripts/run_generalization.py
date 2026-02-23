"""Run held-out template and scale-K generalization evaluations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import (
    run_generalization_templates,
    run_scale_k,
    plot_generalization_templates,
    plot_scale_k,
)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    csv_tpl = os.path.join(results_dir, "metrics_generalization_templates.csv")
    rows_tpl = run_generalization_templates(output_csv=csv_tpl)
    print("Wrote", csv_tpl, "(%d rows)" % len(rows_tpl))
    plot_generalization_templates(
        csv_path=csv_tpl,
        output_path=os.path.join(results_dir, "generalization_templates_plot.png"),
    )

    csv_scale = os.path.join(results_dir, "metrics_scaleK.csv")
    rows_scale = run_scale_k(output_csv=csv_scale)
    print("Wrote", csv_scale, "(%d rows)" % len(rows_scale))
    plot_scale_k(
        csv_path=csv_scale,
        output_path=os.path.join(results_dir, "scaleK_plot.png"),
    )


if __name__ == "__main__":
    main()

