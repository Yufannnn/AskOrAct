#!/usr/bin/env python3
"""Run all three generalization experiments: layout, prior, asymmetric noise."""

import sys
sys.path.insert(0, ".")

from src.eval.run import (
    run_generalization_layout,
    run_generalization_prior,
    run_generalization_asymmetric_noise,
)

if __name__ == "__main__":
    print("=== Generalization: Layout (vertical vs horizontal) ===")
    run_generalization_layout()

    print("\n=== Generalization: Prior (uniform vs distance) ===")
    run_generalization_prior()

    print("\n=== Generalization: Noise (default vs asymmetric) ===")
    run_generalization_asymmetric_noise()

    print("\nAll generalization experiments complete.")
