# Ask or Act for Cooperative Assistance via Inverse Planning

CS6208 project: a cooperative assistant that decides when to ask a clarifying question versus acting immediately to help a principal with a hidden goal under instruction ambiguity.

## Constraints
- No LLMs, embeddings, or heavy NLP.
- Language is a controlled discrete channel (templated instruction → candidate goals).
- Standard Python stack (`numpy`, `matplotlib`); config is code-based.

## Environment Requirements
| Requirement | Version |
|-------------|---------|
| Python | 3.8+ (tested on 3.8.10) |
| OS | Linux/macOS (Windows untested) |
| numpy | ≥1.20 |
| matplotlib | ≥3.3 |
| LaTeX (optional) | TeX Live / MacTeX — only needed to rebuild `progress_report.pdf` |
| CuPy (optional) | GPU backend; set `ASKORACT_USE_GPU=1` to enable |

## Quick Start
```bash
chmod +x run.sh
./run.sh setup       # create .venv and install deps
./run.sh quick-demo  # smoke test: single episode with ASCII render
```

## Full Reproduction (all tables and plots used in reports)
Run these in order from the repo root. Each command is idempotent.

```bash
# 1. Clean any stale artifacts first
rm -f results/*.csv results/*.png results/full_report.md

# 2. Core sweep (results/metrics.csv — 19800 rows)
./run.sh sweep

# 3. Ablations (results/metrics_ablations.csv — 9072 rows)
./run.sh ablations

# 4. Robustness (answer-noise + mismatch; ~8400 rows total)
./run.sh robustness

# 5. Generalization and scale-K (held-out templates + K=5,6)
./run.sh generalization

# 6. Question-difficulty analysis
./run.sh qdifficulty

# 7. Regenerate all plots and full_report.md from the CSVs above
./run.sh report

# 8. (Optional) rebuild progress_report.pdf — requires LaTeX
cd CS6208_Project && latexmk -pdf progress_report.tex && cd ..
```

> **Note:** `./run.sh all` runs steps 2–3–4–7–8(package) only.
> Steps 5 and 6 (`generalization`, `qdifficulty`) must be run separately.
> The main sweep uses 5 core policies at every `K`, plus `easy_info_gain_ask` and `pomcp_planner` only at `K=2`.

## Verification Checks
After a full reproduction run, confirm row counts match:

```bash
python3 - <<'EOF'
import csv, sys, os
expected = {
    "results/metrics.csv":                          19800,
    "results/metrics_ablations.csv":                 9072,
    "results/metrics_robust_answer_noise.csv":       4200,
    "results/metrics_robust_mismatch.csv":           3000,
    "results/metrics_generalization_templates.csv": 14400,
    "results/metrics_scaleK.csv":                    4200,
    "results/metrics_question_difficulty.csv":        800,
}
ok = True
for path, exp in expected.items():
    if not os.path.exists(path):
        print(f"MISSING  {path}")
        ok = False
        continue
    with open(path) as f:
        n = sum(1 for _ in csv.DictReader(f))
    status = "OK" if n == exp else f"MISMATCH (got {n}, expected {exp})"
    print(f"{status:30s} {path}")
    if n != exp:
        ok = False
plots = [
    "results/main_dashboard.png",
    "results/ablations_dashboard.png",
    "results/clarification_quality_entropy_delta.png",
    "results/robust_answer_noise_deltas.png",
    "results/robust_mismatch_deltas.png",
    "results/pareto_K4.png",
]
for p in plots:
    print(("OK" if os.path.exists(p) else "MISSING").ljust(30), p)
    if not os.path.exists(p): ok = False
sys.exit(0 if ok else 1)
EOF
```

## Core Scripts
| Script | Purpose |
|--------|---------|
| `run.sh` | Unified task runner |
| `scripts/quick_demo.py` | Single episode demo |
| `scripts/run_sweep.py` | Full policy sweep over K, eps, beta |
| `scripts/run_ablations.py` | Ablation grid (question cost, wrong-pick mode, deadline margin) |
| `scripts/run_robustness.py` | Answer-noise + principal β-mismatch sweeps |
| `scripts/run_generalization.py` | Held-out templates and scale-K (K=5,6) |
| `scripts/run_question_difficulty.py` | Per-question-type difficulty analysis |
| `scripts/generate_report.py` | Rebuild full_report.md and all plot PNGs |
| `scripts/package_submission.py` | Create results/submission_package.zip |

## Results Artifacts
| File | Rows | Description |
|------|------|-------------|
| `results/metrics.csv` | 19800 | Main sweep — 5 core policies at all K, plus EasyInfoGainAsk and POMCPPlanner at K=2 |
| `results/metrics_ablations.csv` | 9072 | Ablation grid (question cost, wrong-pick, deadline margin) |
| `results/metrics_robust_answer_noise.csv` | 4200 | Answer-noise robustness sweep (K=3,4; all 7 policies) |
| `results/metrics_robust_mismatch.csv` | 3000 | Principal β-mismatch sweep (K=3,4; 5 core policies) |
| `results/metrics_generalization_templates.csv` | 14400 | Held-out instruction template split |
| `results/metrics_scaleK.csv` | 4200 | Scale-K stress test (K up to 6) |
| `results/metrics_question_difficulty.csv` | 800 | Per-question-type difficulty analysis |
| `results/main_dashboard.png` | — | Success / regret / questions by K |
| `results/ablations_dashboard.png` | — | Ablation comparison dashboard |
| `results/clarification_quality_entropy_delta.png` | — | Entropy reduction from first question |
| `results/robust_answer_noise_deltas.png` | — | AskOrAct vs NeverAsk delta across noise levels |
| `results/robust_mismatch_deltas.png` | — | AskOrAct vs NeverAsk delta across β-mismatch |
| `results/pareto_K4.png` | — | Success vs regret Pareto frontier at K=4 |
| `results/full_report.md` | — | Auto-generated full evaluation report |
| `docs/final_report.md` | — | Narrative final report |
| `CS6208_Project/progress_report.pdf` | — | Progress report (submitted March 13 2026) |

## Docs
- `docs/final_report.md` — full narrative with tables and analysis
- `docs/executive_summary.md` — one-page summary
- `docs/presentation_slides.md` — slide outline
- `docs/presentation_talk_track.md` — talk script
- `docs/submission_checklist.md` — submission checklist

## Milestones
| Date | Deliverable |
|------|-------------|
| Feb 13 | Gridworld env, ambiguous instruction templates, scripted principal |
| Feb 20 | Bayesian posterior inference, act-only baseline |
| Feb 27 | Question bank, answer model, ask-or-act VoI policy |
| Mar 13 | Full sweep, ablations, robustness, progress report |
| Apr 10 | Final report, presentation |

## Reproducibility Notes
- All sweeps use deterministic per-episode seeds (`src/eval/run.py`).
- Robustness uses replicate seeds from `config.REPL_SEEDS`.
- 95% bootstrap CIs computed over episodes in `src/eval/plots.py`.
- Seeds are content-hashed from `(base_seed, rep_seed, K, eps, beta, episode_id)` — results are independent of run order.

## Repository Layout
```
src/
  env.py              gridworld environment and step semantics
  config.py           all hyperparameters (single source of truth)
  inference.py        posterior initialisation and Bayesian updates
  backend.py          numpy/CuPy backend selector (GPU-optional)
  agents/
    assistant.py      AskOrAct, NeverAsk, AlwaysAsk, InfoGainAsk, EasyInfoGainAsk, RandomAsk, POMCPPlanner
    principal.py      Boltzmann principal action sampling
  world/
    worldgen.py       procedural world generation
    instructions.py   ambiguous instruction templates
    questions.py      question bank and answer model
  eval/
    run.py            episode loop, sweep runners, ablations
    plots.py          aggregation, dashboards, delta plots
scripts/              entry-point scripts (called by run.sh)
results/              generated CSVs and plots (not committed)
docs/                 narrative reports and presentation materials
CS6208_Project/       LaTeX progress report source
```
