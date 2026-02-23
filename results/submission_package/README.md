# Ask or Act for Cooperative Assistance via Inverse Planning

CS6208 project: a cooperative assistant that decides when to ask a clarifying question versus acting immediately to help a principal with a hidden goal under instruction ambiguity.

## Constraints
- No LLMs, embeddings, or heavy NLP.
- Language is a controlled discrete channel (templated instruction -> candidate goals).
- Standard Python stack (`numpy`, `matplotlib`); config is code-based.

## Quick Start (Linux/macOS)
```bash
chmod +x run.sh
./run.sh setup
./run.sh quick-demo
./run.sh sweep
./run.sh ablations
./run.sh robustness
./run.sh report
./run.sh package
```

## Core Scripts
- `run.sh`: unified task runner for setup/demo/sweep/report/package.
- `scripts/quick_demo.py`: single episode demo with ASCII render.
- `scripts/run_sweep.py`: full policy sweep; writes `results/metrics.csv`.
- `scripts/run_ablations.py`: fixed ablation sweep; writes `results/metrics_ablations.csv`.
- `scripts/run_robustness.py`: focused robustness sweeps (`K=3,4`) and delta plots.
- `scripts/generate_report.py`: rebuilds `results/full_report.md` and dashboards.
- `scripts/package_submission.py`: creates `results/submission_package` and `results/submission_package.zip`.

## Results Artifacts
- Main report: `results/full_report.md`
- Narrative final report: `docs/final_report.md`
- Main dashboard: `results/main_dashboard.png`
- Clarification quality plot: `results/clarification_quality_entropy_delta.png`
- Ablation dashboard: `results/ablations_dashboard.png`
- Main metrics: `results/metrics.csv`
- Ablation metrics: `results/metrics_ablations.csv`
- Robustness metrics: `results/metrics_robust_answer_noise.csv`, `results/metrics_robust_mismatch.csv`
- Robustness plots: `results/robust_answer_noise_deltas.png`, `results/robust_mismatch_deltas.png`

## Presentation and Submission Docs
- Slide outline: `docs/presentation_slides.md`
- Talk track: `docs/presentation_talk_track.md`
- Submission checklist: `docs/submission_checklist.md`

## Milestone Mapping
- Feb 13: gridworld tasks, ambiguous instruction templates, scripted principal.
- Feb 20: posterior inference from instruction + actions, act-only helper baseline.
- Feb 27: question set, answer model, ask-or-act expected-cost policy.
- Mar 13 onward: sweeps, plots, robustness (replicate seeds + CI), ablations, report polish.

## Reproducibility
- Sweep uses deterministic per-episode seeds in `src/eval/run.py`.
- Robustness uses replicate seeds in `src/config.py` (`REPL_SEEDS`).
- Uncertainty uses bootstrap CIs in `src/eval/plots.py`.

## Repository Layout
- `src/env.py`: gridworld environment and step semantics.
- `src/world/`: world generation, instructions, questions/answers.
- `src/inference.py`: posterior initialization and updates.
- `src/agents/assistant.py`: AskOrAct, NeverAsk, AlwaysAsk policies.
- `src/eval/run.py`: episode logic, sweeps, ablations.
- `src/eval/plots.py`: loading, aggregation, dashboards.
