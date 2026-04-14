#!/usr/bin/env python3
"""Run structural OOD, model-mismatch, and failure-penalty experiments."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.eval.run import (
    run_structural_ood,
    run_model_mismatch_extended,
    run_failure_penalty_sweep,
)

if __name__ == "__main__":
    print("=== Structural OOD (grid size + single room) ===")
    run_structural_ood()

    print("\n=== Model-Mismatch Extended (noise + prior) ===")
    run_model_mismatch_extended()

    print("\n=== Failure Penalty Sweep ===")
    run_failure_penalty_sweep()

    print("\nAll new experiments complete.")
