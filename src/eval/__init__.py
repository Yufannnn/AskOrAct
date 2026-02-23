"""Evaluation: run_episode, run_sweep, run_ablations, plots."""

from src.eval.run import run_episode, run_sweep, run_ablations, oracle_steps
from src.eval.plots import (
    load_metrics,
    load_ablation_metrics,
    aggregate_by_condition,
    plot_regret_vs_ambiguity,
    plot_questions_vs_ambiguity,
    plot_success_rate_vs_ambiguity,
    plot_map_rate_vs_ambiguity,
    plot_ablation_figures,
    plot_main_dashboard,
    plot_ablations_dashboard,
)

__all__ = [
    "run_episode",
    "run_sweep",
    "run_ablations",
    "oracle_steps",
    "load_metrics",
    "load_ablation_metrics",
    "aggregate_by_condition",
    "plot_regret_vs_ambiguity",
    "plot_questions_vs_ambiguity",
    "plot_success_rate_vs_ambiguity",
    "plot_map_rate_vs_ambiguity",
    "plot_ablation_figures",
    "plot_main_dashboard",
    "plot_ablations_dashboard",
]
