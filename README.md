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
| LaTeX (optional) | TeX Live / MacTeX — only needed to rebuild `docs/report/final_report.pdf` |
| CuPy (optional) | GPU backend; set `ASKORACT_USE_GPU=1` to enable |

## Quick Start
```bash
chmod +x run.sh
./run.sh setup       # create .venv and install deps
./run.sh quick-demo  # smoke test: single episode with ASCII render
```

## Full Reproduction (all tables and plots used in the report)
```bash
./run.sh sweep            # 1. main sweep → results/metrics/metrics.csv (19,800 rows)
./run.sh ablations        # 2. ablations → results/metrics/metrics_ablations.csv
./run.sh robustness       # 3. answer-noise + β-mismatch sweeps
./run.sh generalization   # 4. held-out templates + scale-K stress test
./run.sh qdifficulty      # 5. per-question-type difficulty
./run.sh report           # 6. regenerate exploratory plots + full_report.md
./run.sh test             # 7. regression tests
```

To rebuild the LaTeX artefacts:
```bash
cd docs/report && pdflatex final_report.tex && bibtex final_report && pdflatex final_report.tex && pdflatex final_report.tex
cd docs/slides && pdflatex presentation_slides.tex
```

## Repository Layout
```
AskOrAct/
├── README.md
├── requirements.txt
├── run.sh                      # unified task runner
├── src/
│   ├── agents/                 # AskOrAct and baseline policies, principal model
│   ├── eval/                   # episode loop, sweeps, plotting
│   ├── world/                  # gridworld, instruction templates, question bank
│   ├── backend.py              # numpy/CuPy backend selector
│   ├── config.py               # all hyperparameters
│   ├── env.py                  # environment and step semantics
│   └── inference.py            # Bayesian posterior updates
├── tests/
│   └── test_regressions.py
├── scripts/
│   ├── sweeps/                 # run_sweep, run_ablations, run_robustness, etc.
│   ├── reporting/              # generate_report.py
│   ├── figures/                # gen_clean_gridworld, build_setup_overview_excalidraw
│   └── demos/                  # quick_demo.py
├── docs/
│   ├── report/                 # final_report.tex/pdf + references.bib
│   ├── slides/                 # presentation_slides.tex/pdf + talk_track.md
│   ├── admin/                  # submission_checklist.md
│   └── archive/                # proposal.tex, progress_report.tex (milestone artefacts)
└── results/
    ├── figures/
    │   ├── paper/              # 8 figures used by final_report.tex + slides
    │   └── exploratory/        # diagnostic plots referenced in appendix / not used
    ├── metrics/                # all sweep CSVs + summary.json
    └── reports/                # auto-generated full_report.md
```

## Core Scripts
| Script | Purpose |
|--------|---------|
| `run.sh` | Unified task runner |
| `scripts/demos/quick_demo.py` | Single episode demo |
| `scripts/sweeps/run_sweep.py` | Full policy sweep over K, ε, β |
| `scripts/sweeps/run_ablations.py` | Ablation grid (question cost, wrong-pick mode, deadline margin) |
| `scripts/sweeps/run_robustness.py` | Answer-noise + principal β-mismatch sweeps |
| `scripts/sweeps/run_generalization.py` | Held-out templates and scale-K (K=5,6) |
| `scripts/sweeps/run_question_difficulty.py` | Per-question-type difficulty analysis |
| `scripts/reporting/generate_report.py` | Rebuild `results/reports/full_report.md` and exploratory plots |
| `scripts/figures/gen_clean_gridworld.py` | Matplotlib maze used inside the Excalidraw setup figure |
| `scripts/figures/build_setup_overview_excalidraw.py` | Build the Excalidraw source for Figure 4 |

## Results Artefacts
| File | Rows | Description |
|------|------|-------------|
| `results/metrics/metrics.csv` | 19,800 | Main sweep — 5 core policies at all K, plus EasyInfoGainAsk and POMCPPlanner at K=2 |
| `results/metrics/metrics_ablations.csv` | 9,072 | Ablation grid |
| `results/metrics/metrics_robust_answer_noise.csv` | 4,200 | Answer-noise robustness sweep |
| `results/metrics/metrics_robust_mismatch.csv` | 3,000 | Principal β-mismatch sweep |
| `results/metrics/metrics_generalization_templates.csv` | 14,400 | Held-out template split |
| `results/metrics/metrics_scaleK.csv` | 4,200 | Scale-K stress test (K up to 6) |
| `results/metrics/metrics_question_difficulty.csv` | 800 | Per-question-type difficulty |
| `results/figures/paper/main_dashboard.png` | — | Success / regret / questions by K |
| `results/figures/paper/beta_inverted_u.png` | — | Non-monotonic assistability in β |
| `results/figures/paper/synergy_decomposition.png` | — | Channel ablation (sub-additive interaction) |
| `results/figures/paper/obs_hurts_mismatch.png` | — | Forced-observation horizon under mismatch |
| `results/reports/full_report.md` | — | Auto-generated diagnostic report |

## Reproducibility Notes
- All sweeps use deterministic per-episode seeds (`src/eval/run.py`).
- Robustness uses replicate seeds from `config.REPL_SEEDS`.
- 95% bootstrap CIs computed over episodes in `src/eval/plots.py`.
- Seeds are content-hashed from `(base_seed, rep_seed, K, ε, β, episode_id)` — results are independent of run order.

## Milestones
| Date | Deliverable |
|------|-------------|
| Feb 13 | Gridworld env, ambiguous instruction templates, scripted principal |
| Feb 20 | Bayesian posterior inference, act-only baseline |
| Feb 27 | Question bank, answer model, ask-or-act VoI policy |
| Mar 13 | Full sweep, ablations, robustness, progress report |
| Apr 11 | Final report, presentation |
