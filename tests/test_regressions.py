import importlib.util
import tempfile
import unittest
from pathlib import Path

from src import config
from src.agents.assistant import _terminal_failure_remaining_cost
from src.eval.run import _policies_for_main_sweep


ROOT = Path(__file__).resolve().parents[1]


def _load_generate_report_module():
    script_path = ROOT / "scripts" / "reporting" / "generate_report.py"
    spec = importlib.util.spec_from_file_location("askoract_generate_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentScheduleTests(unittest.TestCase):
    def test_main_sweep_policy_schedule_matches_expected_row_count(self):
        per_condition = len(config.REPL_SEEDS) * config.N_EPISODES_PER_SEED
        total_rows = 0
        for ambiguity_k in config.AMBIGUITY_LEVELS:
            total_rows += (
                len(_policies_for_main_sweep(ambiguity_k))
                * len(config.EPS_LEVELS)
                * len(config.BETA_LEVELS)
                * per_condition
            )
        self.assertEqual(total_rows, 19800)

    def test_robust_mismatch_policy_schedule_matches_expected_row_count(self):
        total_rows = (
            2  # K in {3,4}
            * 3  # principal_beta in {1,2,4}
            * len(config.ROBUST_MISMATCH_POLICIES)
            * len(config.REPL_SEEDS)
            * config.N_EPISODES_PER_SEED
        )
        self.assertEqual(total_rows, 3000)


class PlannerCostTests(unittest.TestCase):
    def test_terminal_failure_remaining_cost_matches_eval_objective(self):
        question_cost_total = 1.5
        actual_steps_before_failure = 3
        episode_max_steps = 10
        total_failure_cost = (
            actual_steps_before_failure
            + question_cost_total
            + _terminal_failure_remaining_cost(actual_steps_before_failure, episode_max_steps)
        )
        self.assertEqual(total_failure_cost, episode_max_steps + question_cost_total)

    def test_terminal_failure_remaining_cost_is_zero_at_deadline(self):
        self.assertEqual(_terminal_failure_remaining_cost(10, 10), 0.0)


class ReportRegressionTests(unittest.TestCase):
    def test_report_does_not_backfill_main_sweep_contrasts_from_scalek(self):
        module = _load_generate_report_module()
        metrics_dir = ROOT / "results" / "metrics"
        module.CSV_PATH = str(metrics_dir / "metrics.csv")
        module.ABLATION_CSV_PATH = str(metrics_dir / "metrics_ablations.csv")
        module.ROBUST_ANSWER_CSV_PATH = str(metrics_dir / "metrics_robust_answer_noise.csv")
        module.ROBUST_MISMATCH_CSV_PATH = str(metrics_dir / "metrics_robust_mismatch.csv")
        module.GENERALIZATION_TEMPLATES_CSV_PATH = str(metrics_dir / "metrics_generalization_templates.csv")
        module.SCALEK_CSV_PATH = str(metrics_dir / "metrics_scaleK.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            module.REPORT_MD = str(Path(tmpdir) / "full_report.md")
            module.generate_report()
            text = Path(module.REPORT_MD).read_text(encoding="utf-8")

        self.assertIn(
            "| DeltaSuccess (AskOrAct - EasyInfoGainAsk) | Not evaluated in main sweep at K=3,4 | 0 |",
            text,
        )
        self.assertIn(
            "| DeltaSuccess (AskOrAct - POMCP) | Not evaluated in main sweep at K=3,4 | 0 |",
            text,
        )
        self.assertNotIn("| DeltaSuccess (AskOrAct - EasyInfoGainAsk) | -1.50%", text)


if __name__ == "__main__":
    unittest.main()
