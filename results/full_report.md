# AskOrAct - Full Evaluation Report

**Generated:** 2026-02-23 18:00

## Setup

- **Principal cannot pick objects** (`PRINCIPAL_CAN_PICK = False`).
- **Success requires assistant to pick the true goal object.**
- **Episode deadline:** per-episode max steps = oracle shortest assistant path + margin.
- **Cost rule:** `team_cost = steps + QUESTION_COST * questions_asked`; failed episodes use `team_cost = episode_max_steps + QUESTION_COST * questions_asked`.
- **Policies:** AskOrAct, NeverAsk, AlwaysAsk.
- **Conditions:** Ambiguity K in {1, 2, 3, 4}, eps in {0.0, 0.05, 0.1}, beta in {1.0, 2.0, 4.0}.
- **Replicate seeds:** [0, 1, 2, 3, 4].
- **Episodes per (policy, K, eps, beta):** 100 (5 reps x 20 eps/rep).
- **Uncertainty:** 95% bootstrap CI over episodes (B=1000).

---

## Summary by condition (K, eps, beta, policy)

| K | eps | beta | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |
|---|-----|------|--------|--------------|-----------|---------------|------------|-------------|
| 1 | 0.0 | 1.0 | always_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | ask_or_act | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | never_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 2.0 | always_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | ask_or_act | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | never_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 4.0 | always_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | ask_or_act | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | never_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.05 | 1.0 | always_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | ask_or_act | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | never_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 2.0 | always_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | ask_or_act | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | never_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 4.0 | always_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | ask_or_act | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | never_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.1 | 1.0 | always_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | ask_or_act | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | never_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 2.0 | always_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | ask_or_act | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | never_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 4.0 | always_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | ask_or_act | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | never_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 2 | 0.0 | 1.0 | always_ask | 100.00% | 8.7 | 1.72 | 3.18 | 100.00% |
| 2 | 0.0 | 1.0 | ask_or_act | 85.00% | 8.5 | 0.00 | 2.16 | 95.00% |
| 2 | 0.0 | 1.0 | never_ask | 85.00% | 8.5 | 0.00 | 2.16 | 95.00% |
| 2 | 0.0 | 2.0 | always_ask | 100.00% | 8.8 | 1.78 | 3.19 | 100.00% |
| 2 | 0.0 | 2.0 | ask_or_act | 94.00% | 7.9 | 0.00 | 1.46 | 98.00% |
| 2 | 0.0 | 2.0 | never_ask | 94.00% | 7.9 | 0.00 | 1.46 | 98.00% |
| 2 | 0.0 | 4.0 | always_ask | 100.00% | 9.0 | 1.83 | 3.31 | 100.00% |
| 2 | 0.0 | 4.0 | ask_or_act | 98.00% | 7.9 | 0.00 | 1.26 | 98.00% |
| 2 | 0.0 | 4.0 | never_ask | 98.00% | 7.9 | 0.00 | 1.26 | 98.00% |
| 2 | 0.05 | 1.0 | always_ask | 100.00% | 8.8 | 1.70 | 3.01 | 100.00% |
| 2 | 0.05 | 1.0 | ask_or_act | 84.00% | 8.8 | 0.00 | 2.15 | 91.00% |
| 2 | 0.05 | 1.0 | never_ask | 84.00% | 8.8 | 0.00 | 2.15 | 91.00% |
| 2 | 0.05 | 2.0 | always_ask | 100.00% | 8.7 | 1.75 | 3.08 | 100.00% |
| 2 | 0.05 | 2.0 | ask_or_act | 90.00% | 8.2 | 0.00 | 1.69 | 98.00% |
| 2 | 0.05 | 2.0 | never_ask | 90.00% | 8.2 | 0.00 | 1.69 | 98.00% |
| 2 | 0.05 | 4.0 | always_ask | 100.00% | 8.5 | 1.67 | 2.98 | 100.00% |
| 2 | 0.05 | 4.0 | ask_or_act | 93.00% | 7.8 | 0.00 | 1.49 | 98.00% |
| 2 | 0.05 | 4.0 | never_ask | 93.00% | 7.8 | 0.00 | 1.49 | 98.00% |
| 2 | 0.1 | 1.0 | always_ask | 100.00% | 8.3 | 1.67 | 3.04 | 100.00% |
| 2 | 0.1 | 1.0 | ask_or_act | 85.00% | 8.0 | 0.00 | 1.92 | 96.00% |
| 2 | 0.1 | 1.0 | never_ask | 85.00% | 8.0 | 0.00 | 1.92 | 96.00% |
| 2 | 0.1 | 2.0 | always_ask | 100.00% | 8.4 | 1.48 | 2.69 | 100.00% |
| 2 | 0.1 | 2.0 | ask_or_act | 96.00% | 8.0 | 0.00 | 1.56 | 97.00% |
| 2 | 0.1 | 2.0 | never_ask | 96.00% | 8.0 | 0.00 | 1.56 | 97.00% |
| 2 | 0.1 | 4.0 | always_ask | 100.00% | 8.2 | 1.85 | 3.17 | 100.00% |
| 2 | 0.1 | 4.0 | ask_or_act | 95.00% | 7.3 | 0.00 | 1.32 | 98.00% |
| 2 | 0.1 | 4.0 | never_ask | 95.00% | 7.3 | 0.00 | 1.32 | 98.00% |
| 3 | 0.0 | 1.0 | always_ask | 87.00% | 9.2 | 2.41 | 4.14 | 98.00% |
| 3 | 0.0 | 1.0 | ask_or_act | 90.00% | 8.1 | 0.74 | 2.19 | 95.00% |
| 3 | 0.0 | 1.0 | never_ask | 76.00% | 8.8 | 0.00 | 2.51 | 91.00% |
| 3 | 0.0 | 2.0 | always_ask | 89.00% | 9.1 | 2.28 | 3.99 | 99.00% |
| 3 | 0.0 | 2.0 | ask_or_act | 90.00% | 8.1 | 0.66 | 2.19 | 99.00% |
| 3 | 0.0 | 2.0 | never_ask | 91.00% | 8.2 | 0.00 | 2.02 | 98.00% |
| 3 | 0.0 | 4.0 | always_ask | 94.00% | 9.3 | 2.41 | 4.14 | 99.00% |
| 3 | 0.0 | 4.0 | ask_or_act | 93.00% | 7.9 | 0.55 | 1.86 | 100.00% |
| 3 | 0.0 | 4.0 | never_ask | 85.00% | 8.3 | 0.00 | 1.94 | 98.00% |
| 3 | 0.05 | 1.0 | always_ask | 87.00% | 9.7 | 2.28 | 4.19 | 98.00% |
| 3 | 0.05 | 1.0 | ask_or_act | 86.00% | 8.7 | 0.76 | 2.47 | 91.00% |
| 3 | 0.05 | 1.0 | never_ask | 76.00% | 9.2 | 0.00 | 2.59 | 83.00% |
| 3 | 0.05 | 2.0 | always_ask | 88.00% | 10.2 | 2.36 | 4.34 | 98.00% |
| 3 | 0.05 | 2.0 | ask_or_act | 91.00% | 9.0 | 0.61 | 2.23 | 96.00% |
| 3 | 0.05 | 2.0 | never_ask | 84.00% | 9.2 | 0.00 | 2.20 | 96.00% |
| 3 | 0.05 | 4.0 | always_ask | 91.00% | 9.2 | 2.18 | 3.89 | 98.00% |
| 3 | 0.05 | 4.0 | ask_or_act | 93.00% | 8.2 | 0.65 | 2.08 | 98.00% |
| 3 | 0.05 | 4.0 | never_ask | 91.00% | 8.2 | 0.00 | 1.75 | 98.00% |
| 3 | 0.1 | 1.0 | always_ask | 91.00% | 9.2 | 2.12 | 3.85 | 95.00% |
| 3 | 0.1 | 1.0 | ask_or_act | 94.00% | 8.2 | 0.76 | 2.18 | 98.00% |
| 3 | 0.1 | 1.0 | never_ask | 78.00% | 8.7 | 0.00 | 2.34 | 94.00% |
| 3 | 0.1 | 2.0 | always_ask | 94.00% | 9.5 | 2.34 | 4.11 | 98.00% |
| 3 | 0.1 | 2.0 | ask_or_act | 94.00% | 8.2 | 0.61 | 1.97 | 98.00% |
| 3 | 0.1 | 2.0 | never_ask | 88.00% | 8.4 | 0.00 | 1.84 | 95.00% |
| 3 | 0.1 | 4.0 | always_ask | 91.00% | 9.3 | 2.26 | 4.02 | 98.00% |
| 3 | 0.1 | 4.0 | ask_or_act | 94.00% | 8.2 | 0.65 | 2.08 | 96.00% |
| 3 | 0.1 | 4.0 | never_ask | 85.00% | 8.6 | 0.00 | 2.15 | 95.00% |
| 4 | 0.0 | 1.0 | always_ask | 83.00% | 10.0 | 2.50 | 4.47 | 95.00% |
| 4 | 0.0 | 1.0 | ask_or_act | 80.00% | 9.3 | 0.68 | 2.84 | 91.00% |
| 4 | 0.0 | 1.0 | never_ask | 66.00% | 9.6 | 0.00 | 2.78 | 92.00% |
| 4 | 0.0 | 2.0 | always_ask | 92.00% | 10.1 | 2.62 | 4.53 | 98.00% |
| 4 | 0.0 | 2.0 | ask_or_act | 93.00% | 8.6 | 0.47 | 1.92 | 96.00% |
| 4 | 0.0 | 2.0 | never_ask | 86.00% | 9.0 | 0.00 | 2.08 | 97.00% |
| 4 | 0.0 | 4.0 | always_ask | 94.00% | 9.9 | 2.52 | 4.43 | 100.00% |
| 4 | 0.0 | 4.0 | ask_or_act | 95.00% | 8.4 | 0.50 | 1.89 | 99.00% |
| 4 | 0.0 | 4.0 | never_ask | 90.00% | 8.5 | 0.00 | 1.81 | 99.00% |
| 4 | 0.05 | 1.0 | always_ask | 84.00% | 10.1 | 2.59 | 4.63 | 98.00% |
| 4 | 0.05 | 1.0 | ask_or_act | 77.00% | 9.2 | 0.74 | 2.76 | 95.00% |
| 4 | 0.05 | 1.0 | never_ask | 68.00% | 9.6 | 0.00 | 2.78 | 85.00% |
| 4 | 0.05 | 2.0 | always_ask | 89.00% | 10.1 | 2.51 | 4.46 | 98.00% |
| 4 | 0.05 | 2.0 | ask_or_act | 89.00% | 8.9 | 0.56 | 2.32 | 100.00% |
| 4 | 0.05 | 2.0 | never_ask | 83.00% | 9.1 | 0.00 | 2.23 | 95.00% |
| 4 | 0.05 | 4.0 | always_ask | 90.00% | 9.8 | 2.62 | 4.62 | 100.00% |
| 4 | 0.05 | 4.0 | ask_or_act | 96.00% | 8.4 | 0.52 | 2.22 | 98.00% |
| 4 | 0.05 | 4.0 | never_ask | 95.00% | 8.4 | 0.00 | 1.98 | 98.00% |
| 4 | 0.1 | 1.0 | always_ask | 77.00% | 9.5 | 2.64 | 4.74 | 96.00% |
| 4 | 0.1 | 1.0 | ask_or_act | 77.00% | 8.7 | 0.85 | 3.00 | 96.00% |
| 4 | 0.1 | 1.0 | never_ask | 60.00% | 9.3 | 0.00 | 3.21 | 88.00% |
| 4 | 0.1 | 2.0 | always_ask | 85.00% | 9.5 | 2.56 | 4.53 | 100.00% |
| 4 | 0.1 | 2.0 | ask_or_act | 88.00% | 8.3 | 0.58 | 2.34 | 97.00% |
| 4 | 0.1 | 2.0 | never_ask | 76.00% | 8.7 | 0.00 | 2.45 | 93.00% |
| 4 | 0.1 | 4.0 | always_ask | 84.00% | 9.6 | 2.62 | 4.64 | 97.00% |
| 4 | 0.1 | 4.0 | ask_or_act | 91.00% | 8.4 | 0.60 | 2.48 | 97.00% |
| 4 | 0.1 | 4.0 | never_ask | 86.00% | 8.6 | 0.00 | 2.32 | 97.00% |

---

## Aggregate by K (averaged over eps and beta)

| K | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |
|---|--------|--------------|-----------|---------------|------------|-------------|
| 1 | always_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | ask_or_act | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | never_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 2 | always_ask | 100.00% | 8.6 | 1.72 | 3.07 | 100.00% |
| 2 | ask_or_act | 91.11% | 8.1 | 0.00 | 1.67 | 96.56% |
| 2 | never_ask | 91.11% | 8.1 | 0.00 | 1.67 | 96.56% |
| 3 | always_ask | 90.22% | 9.4 | 2.29 | 4.08 | 97.89% |
| 3 | ask_or_act | 91.67% | 8.3 | 0.67 | 2.14 | 96.78% |
| 3 | never_ask | 83.78% | 8.6 | 0.00 | 2.15 | 94.22% |
| 4 | always_ask | 86.44% | 9.9 | 2.58 | 4.56 | 98.00% |
| 4 | ask_or_act | 87.33% | 8.7 | 0.61 | 2.42 | 96.56% |
| 4 | never_ask | 78.89% | 9.0 | 0.00 | 2.40 | 93.78% |

---

## High-Ambiguity Snapshot (K=3,4)

- **always_ask:** success 88.33%, avg regret 4.32, avg questions 2.43.
- **ask_or_act:** success 89.50%, avg regret 2.28, avg questions 0.64.
- **never_ask:** success 81.33%, avg regret 2.28, avg questions 0.00.

## Robustness Summary (K=3,4)

All results are averaged over replicate seeds; uncertainty is 95% bootstrap CI over episodes.

| K | Policy | Success (mean [95% CI]) | Regret (mean [95% CI]) | Questions (mean [95% CI]) |
|---|--------|--------------------------|-------------------------|----------------------------|
| 3 | always_ask | 90.22% [88.33%, 92.11%] | 4.08 [3.97, 4.19] | 2.29 [2.23, 2.35] |
| 3 | ask_or_act | 91.67% [89.78%, 93.44%] | 2.14 [2.05, 2.25] | 0.67 [0.63, 0.70] |
| 3 | never_ask | 83.78% [81.22%, 86.22%] | 2.15 [2.02, 2.27] | 0.00 [0.00, 0.00] |
| 4 | always_ask | 86.44% [84.00%, 88.34%] | 4.56 [4.46, 4.66] | 2.58 [2.52, 2.62] |
| 4 | ask_or_act | 87.33% [85.11%, 89.34%] | 2.42 [2.30, 2.53] | 0.61 [0.58, 0.64] |
| 4 | never_ask | 78.89% [75.89%, 81.34%] | 2.40 [2.28, 2.53] | 0.00 [0.00, 0.00] |

## Policy Difference CIs (K=3,4)

Deltas are bootstrap-estimated on paired episode keys: (K, eps, beta, rep_seed, episode_id).

| Contrast | Delta mean [95% CI] | N paired episodes |
|----------|-----------------------|-------------------|
| DeltaSuccess (AskOrAct - NeverAsk) | 8.17% [6.61%, 9.78%] | 1800 |
| DeltaRegret (AskOrAct - AlwaysAsk) | -2.04 [-2.13, -1.95] | 1800 |

## Failure Mode Breakdown

| K | Policy | Failure rate | failure_by_timeout rate | failure_by_wrong_pick rate |
|---|--------|--------------|-------------------------|----------------------------|
| 3 | always_ask | 9.78% | 9.78% | 0.00% |
| 3 | ask_or_act | 8.33% | 8.33% | 0.00% |
| 3 | never_ask | 16.22% | 16.22% | 0.00% |
| 4 | always_ask | 13.56% | 13.56% | 0.00% |
| 4 | ask_or_act | 12.67% | 12.67% | 0.00% |
| 4 | never_ask | 21.11% | 21.11% | 0.00% |

## Wrong-Pick Fail Semantics

- Main sweep keeps `WRONG_PICK_FAIL = False` for comparability.

## Ablation Notes

- **Mode A (time-only):** ASK counts as step, QUESTION_COST = 0.0.
- **Mode B (comm-cost-only):** ASK does not count as step, QUESTION_COST = 0.5.
- **Mode C (both):** ASK counts as step, QUESTION_COST = 0.5.
- This separates temporal cost from communication cost.

- **modeA (K=3,4, wrong_pick_fail=False):**
  ask_or_act -> success 89.58%, regret 1.72, questions 0.62
  never_ask -> success 85.42%, regret 2.16, questions 0.00
  always_ask -> success 81.25%, regret 2.85, questions 2.39
- **modeB (K=3,4, wrong_pick_fail=False):**
  ask_or_act -> success 90.97%, regret 1.66, questions 0.65
  never_ask -> success 82.64%, regret 2.22, questions 0.00
  always_ask -> success 95.83%, regret 2.03, questions 2.44
- **modeC (K=3,4, wrong_pick_fail=False):**
  ask_or_act -> success 87.50%, regret 2.37, questions 0.67
  never_ask -> success 84.03%, regret 1.98, questions 0.00
  always_ask -> success 78.47%, regret 4.36, questions 2.51

---

*Raw data: `results/metrics.csv`*
*Main plots: `results/main_dashboard.png`*
*Ablations: `results/metrics_ablations.csv` and `results/ablations_dashboard.png`*
