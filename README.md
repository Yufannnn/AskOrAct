# Ask or Act for Cooperative Assistance via Inverse Planning

CS6208 course project: a small cooperative decision-making system where an assistant decides when to **ask** a clarifying question vs **act** immediately to help a principal complete a hidden goal under ambiguity.

## Constraints

- No LLMs, embeddings, or heavy NLP. Language is a controlled discrete channel (short instruction templates → candidate goals).
- Standard Python + numpy + matplotlib (tqdm optional). Hardcoded config in code; no CLI args.

## Setup

**Windows:** Run **`setup.bat`** once (creates `.venv`, installs deps). Then **`run_demo.bat`** or **`run_report.bat`** (they call `setup.bat` if `.venv` is missing).

**Or** with Python on PATH: `pip install -r requirements.txt`, then run scripts with `python scripts/...`.

Optional GPU (e.g. laptop with GTX 4060): install CuPy for your CUDA version, then enable with:

```bash
set ASKORACT_USE_GPU=1
python scripts/run_sweep.py
```

(On Linux/macOS use `export ASKORACT_USE_GPU=1`.) The core loop stays on CPU; the backend is ready for GPU-accelerated batch ops if you add them later.

## Project milestones (proposal)

| Milestone | Deliverable | Code |
|-----------|-------------|------|
| **Feb 13** | Gridworld tasks, goal sets, ambiguous instruction templates, scripted principal | `src/env.py`, `src/world/worldgen.py`, `src/world/instructions.py`, `src/agents/principal.py` |
| **Feb 20** | Posterior inference from actions + instruction; act-only helper policy | `src/inference.py`, `src/agents/assistant.py` (policy_never_ask) |

To run **only** the Feb 13 + Feb 20 stack (no questions, act-only assistant):

```bash
# Single demo (one episode, printed to console)
python scripts/run_milestone_feb13_feb20.py

# Run experiments and generate report (metrics + Markdown report)
python scripts/run_milestone_feb13_feb20.py --report
```

With `--report`, the script runs multiple episodes across ambiguity K ∈ {1,2,3,4} and principal noise eps ∈ {0.0, 0.05, 0.1}, writes **results/milestone_feb13_feb20_metrics.csv**, **results/milestone_feb13_feb20_summary.json**, and **results/milestone_feb13_feb20_report.md** (summary table and aggregate stats for the progress report).

## Run quick demo

One episode with ASCII grid printed each step (full system: includes ask-or-act and questions):

```bash
python scripts/quick_demo.py
```

## Run experiments (sweep)

Evaluate policies over ambiguity K ∈ {1,2,3,4}, eps ∈ {0.0, 0.05, 0.1}, beta ∈ {1.0, 2.0, 4.0}. Writes `results/metrics.csv`, `results/summary.json`, and plots:

```bash
python scripts/run_sweep.py
```

Outputs:

- **results/metrics.csv** — per-episode metrics (success, steps, questions_asked, regret)
- **results/summary.json** — aggregated success_rate, avg_steps, avg_questions, avg_regret per condition
- **results/regret_vs_ambiguity.png** — regret vs K
- **results/questions_vs_ambiguity.png** — questions asked vs K

To use multiple CPU cores for the sweep, set `N_WORKERS` in `src/config.py` (e.g. `N_WORKERS = 4`).

## Code layout (refactored)

| Path | Purpose |
|------|--------|
| **Config & backend** | |
| `src/config.py` | All constants (GPU, env, principal, sweep); edit here or use env var `ASKORACT_USE_GPU` |
| `src/backend.py` | Optional GPU backend (numpy / CuPy); `xp()`, `device()` |
| **Env** | |
| `src/env.py` | `GridWorldEnv`, `grid_distance`, actions, object attributes |
| **World** | |
| `src/world/` | World generation, instructions, questions |
| `src/world/worldgen.py` | `generate_world(seed, N, M, ambiguity_K)` |
| `src/world/instructions.py` | Templates, `instruction_to_candidate_goals(u, world)` |
| `src/world/questions.py` | Question menu, `answer_question`, `answer_likelihood` |
| **Agents** | |
| `src/agents/` | Principal and assistant policies |
| `src/agents/principal.py` | `principal_action_probs`, `sample_principal_action` |
| `src/agents/assistant.py` | AskOrAct, NeverAsk, AlwaysAsk; BFS and task action |
| **Inference** | |
| `src/inference.py` | `init_posterior`, `update_posterior`, `posterior_entropy` |
| **Evaluation** | |
| `src/eval/run.py` | `run_episode`, `run_sweep`, `oracle_steps` (optional parallel workers) |
| `src/eval/plots.py` | `load_metrics`, `plot_regret_vs_ambiguity`, `plot_questions_vs_ambiguity` |
| **Scripts** | |
| `scripts/run_milestone_feb13_feb20.py` | Feb 13+20 only: gridworld, principal, posterior, act-only assistant (no questions) |
| `scripts/quick_demo.py` | Single episode with render (full system) |
| `scripts/run_sweep.py` | Full sweep and save results + plots |
