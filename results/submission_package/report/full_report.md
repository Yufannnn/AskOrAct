# AskOrAct - Full Evaluation Report

**Generated:** 2026-02-23 18:58

## Setup

- **Principal cannot pick objects** (`PRINCIPAL_CAN_PICK = False`).
- **Success requires assistant to pick the true goal object.**
- **Episode deadline:** per-episode max steps = oracle shortest assistant path + margin.
- **Cost rule:** `team_cost = steps + QUESTION_COST * questions_asked`; failed episodes use `team_cost = episode_max_steps + QUESTION_COST * questions_asked`.
- **Policies:** AskOrAct, NeverAsk, AlwaysAsk, InfoGainAsk, RandomAsk.
- **Conditions:** Ambiguity K in {1, 2, 3, 4}, eps in {0.0, 0.05, 0.1}, beta in {1.0, 2.0, 4.0}.
- **Replicate seeds:** [0, 1, 2, 3, 4].
- **Episodes per (policy, K, eps, beta):** 100 (5 reps x 20 eps/rep).
- **Uncertainty:** 95% bootstrap CI over episodes (B=1000).

## Baselines

- **info_gain_ask:** computes expected entropy reduction `IG(q)=H(b)-E[H|q]` for each question and asks `argmax_q IG(q)` when IG passes threshold (and optional entropy gate), otherwise acts toward MAP goal.
- **random_ask:** uses the same ask gating as `info_gain_ask` but picks a random available question when asking.

---

## Summary by condition (K, eps, beta, policy)

| K | eps | beta | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |
|---|-----|------|--------|--------------|-----------|---------------|------------|-------------|
| 1 | 0.0 | 1.0 | always_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | ask_or_act | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | info_gain_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | never_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 1.0 | random_ask | 100.00% | 6.7 | 0.00 | 0.58 | 100.00% |
| 1 | 0.0 | 2.0 | always_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | ask_or_act | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | info_gain_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | never_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 2.0 | random_ask | 100.00% | 6.6 | 0.00 | 0.55 | 100.00% |
| 1 | 0.0 | 4.0 | always_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | ask_or_act | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | info_gain_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | never_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.0 | 4.0 | random_ask | 100.00% | 7.0 | 0.00 | 0.53 | 100.00% |
| 1 | 0.05 | 1.0 | always_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | ask_or_act | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | info_gain_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | never_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 1.0 | random_ask | 100.00% | 6.9 | 0.00 | 0.57 | 100.00% |
| 1 | 0.05 | 2.0 | always_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | ask_or_act | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | info_gain_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | never_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 2.0 | random_ask | 100.00% | 7.0 | 0.00 | 0.49 | 100.00% |
| 1 | 0.05 | 4.0 | always_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | ask_or_act | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | info_gain_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | never_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.05 | 4.0 | random_ask | 100.00% | 7.3 | 0.00 | 0.54 | 100.00% |
| 1 | 0.1 | 1.0 | always_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | ask_or_act | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | info_gain_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | never_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 1.0 | random_ask | 100.00% | 7.2 | 0.00 | 0.59 | 100.00% |
| 1 | 0.1 | 2.0 | always_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | ask_or_act | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | info_gain_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | never_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 2.0 | random_ask | 100.00% | 6.2 | 0.00 | 0.46 | 100.00% |
| 1 | 0.1 | 4.0 | always_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | ask_or_act | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | info_gain_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | never_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 1 | 0.1 | 4.0 | random_ask | 100.00% | 6.1 | 0.00 | 0.42 | 100.00% |
| 2 | 0.0 | 1.0 | always_ask | 100.00% | 8.7 | 1.72 | 3.18 | 100.00% |
| 2 | 0.0 | 1.0 | ask_or_act | 85.00% | 8.5 | 0.00 | 2.16 | 95.00% |
| 2 | 0.0 | 1.0 | info_gain_ask | 100.00% | 7.9 | 1.00 | 2.10 | 100.00% |
| 2 | 0.0 | 1.0 | never_ask | 85.00% | 8.5 | 0.00 | 2.16 | 95.00% |
| 2 | 0.0 | 1.0 | random_ask | 100.00% | 8.7 | 1.73 | 3.19 | 100.00% |
| 2 | 0.0 | 2.0 | always_ask | 100.00% | 8.8 | 1.78 | 3.19 | 100.00% |
| 2 | 0.0 | 2.0 | ask_or_act | 94.00% | 7.9 | 0.00 | 1.46 | 98.00% |
| 2 | 0.0 | 2.0 | info_gain_ask | 100.00% | 8.0 | 1.00 | 2.02 | 100.00% |
| 2 | 0.0 | 2.0 | never_ask | 94.00% | 7.9 | 0.00 | 1.46 | 98.00% |
| 2 | 0.0 | 2.0 | random_ask | 100.00% | 8.8 | 1.84 | 3.28 | 100.00% |
| 2 | 0.0 | 4.0 | always_ask | 100.00% | 9.0 | 1.83 | 3.31 | 100.00% |
| 2 | 0.0 | 4.0 | ask_or_act | 98.00% | 7.9 | 0.00 | 1.26 | 98.00% |
| 2 | 0.0 | 4.0 | info_gain_ask | 100.00% | 8.2 | 1.00 | 2.06 | 100.00% |
| 2 | 0.0 | 4.0 | never_ask | 98.00% | 7.9 | 0.00 | 1.26 | 98.00% |
| 2 | 0.0 | 4.0 | random_ask | 100.00% | 8.9 | 1.66 | 3.05 | 100.00% |
| 2 | 0.05 | 1.0 | always_ask | 100.00% | 8.8 | 1.70 | 3.01 | 100.00% |
| 2 | 0.05 | 1.0 | ask_or_act | 84.00% | 8.8 | 0.00 | 2.15 | 91.00% |
| 2 | 0.05 | 1.0 | info_gain_ask | 100.00% | 8.1 | 1.00 | 1.96 | 100.00% |
| 2 | 0.05 | 1.0 | never_ask | 84.00% | 8.8 | 0.00 | 2.15 | 91.00% |
| 2 | 0.05 | 1.0 | random_ask | 100.00% | 8.8 | 1.72 | 3.04 | 100.00% |
| 2 | 0.05 | 2.0 | always_ask | 100.00% | 8.7 | 1.75 | 3.08 | 100.00% |
| 2 | 0.05 | 2.0 | ask_or_act | 90.00% | 8.2 | 0.00 | 1.69 | 98.00% |
| 2 | 0.05 | 2.0 | info_gain_ask | 100.00% | 8.0 | 1.00 | 1.95 | 100.00% |
| 2 | 0.05 | 2.0 | never_ask | 90.00% | 8.2 | 0.00 | 1.69 | 98.00% |
| 2 | 0.05 | 2.0 | random_ask | 100.00% | 8.6 | 1.63 | 2.90 | 100.00% |
| 2 | 0.05 | 4.0 | always_ask | 100.00% | 8.5 | 1.67 | 2.98 | 100.00% |
| 2 | 0.05 | 4.0 | ask_or_act | 93.00% | 7.8 | 0.00 | 1.49 | 98.00% |
| 2 | 0.05 | 4.0 | info_gain_ask | 100.00% | 7.8 | 1.00 | 1.98 | 100.00% |
| 2 | 0.05 | 4.0 | never_ask | 93.00% | 7.8 | 0.00 | 1.49 | 98.00% |
| 2 | 0.05 | 4.0 | random_ask | 100.00% | 8.6 | 1.76 | 3.12 | 100.00% |
| 2 | 0.1 | 1.0 | always_ask | 100.00% | 8.3 | 1.67 | 3.04 | 100.00% |
| 2 | 0.1 | 1.0 | ask_or_act | 85.00% | 8.0 | 0.00 | 1.92 | 96.00% |
| 2 | 0.1 | 1.0 | info_gain_ask | 100.00% | 7.7 | 1.00 | 2.04 | 100.00% |
| 2 | 0.1 | 1.0 | never_ask | 85.00% | 8.0 | 0.00 | 1.92 | 96.00% |
| 2 | 0.1 | 1.0 | random_ask | 100.00% | 8.4 | 1.71 | 3.10 | 100.00% |
| 2 | 0.1 | 2.0 | always_ask | 100.00% | 8.4 | 1.48 | 2.69 | 100.00% |
| 2 | 0.1 | 2.0 | ask_or_act | 96.00% | 8.0 | 0.00 | 1.56 | 97.00% |
| 2 | 0.1 | 2.0 | info_gain_ask | 100.00% | 8.0 | 1.00 | 1.97 | 100.00% |
| 2 | 0.1 | 2.0 | never_ask | 96.00% | 8.0 | 0.00 | 1.56 | 97.00% |
| 2 | 0.1 | 2.0 | random_ask | 100.00% | 8.6 | 1.67 | 2.98 | 100.00% |
| 2 | 0.1 | 4.0 | always_ask | 100.00% | 8.2 | 1.85 | 3.17 | 100.00% |
| 2 | 0.1 | 4.0 | ask_or_act | 95.00% | 7.3 | 0.00 | 1.32 | 98.00% |
| 2 | 0.1 | 4.0 | info_gain_ask | 100.00% | 7.4 | 1.00 | 1.90 | 100.00% |
| 2 | 0.1 | 4.0 | never_ask | 95.00% | 7.3 | 0.00 | 1.32 | 98.00% |
| 2 | 0.1 | 4.0 | random_ask | 99.00% | 8.1 | 1.73 | 2.98 | 100.00% |
| 3 | 0.0 | 1.0 | always_ask | 87.00% | 9.2 | 2.41 | 4.14 | 98.00% |
| 3 | 0.0 | 1.0 | ask_or_act | 90.00% | 8.1 | 0.74 | 2.19 | 95.00% |
| 3 | 0.0 | 1.0 | info_gain_ask | 93.00% | 8.2 | 1.25 | 2.58 | 98.00% |
| 3 | 0.0 | 1.0 | never_ask | 76.00% | 8.8 | 0.00 | 2.51 | 91.00% |
| 3 | 0.0 | 1.0 | random_ask | 92.00% | 8.8 | 1.96 | 3.56 | 100.00% |
| 3 | 0.0 | 2.0 | always_ask | 89.00% | 9.1 | 2.28 | 3.99 | 99.00% |
| 3 | 0.0 | 2.0 | ask_or_act | 90.00% | 8.1 | 0.66 | 2.19 | 99.00% |
| 3 | 0.0 | 2.0 | info_gain_ask | 92.00% | 8.2 | 1.23 | 2.60 | 100.00% |
| 3 | 0.0 | 2.0 | never_ask | 91.00% | 8.2 | 0.00 | 2.02 | 98.00% |
| 3 | 0.0 | 2.0 | random_ask | 90.00% | 8.8 | 1.87 | 3.52 | 100.00% |
| 3 | 0.0 | 4.0 | always_ask | 94.00% | 9.3 | 2.41 | 4.14 | 99.00% |
| 3 | 0.0 | 4.0 | ask_or_act | 93.00% | 7.9 | 0.55 | 1.86 | 100.00% |
| 3 | 0.0 | 4.0 | info_gain_ask | 98.00% | 8.1 | 1.20 | 2.37 | 100.00% |
| 3 | 0.0 | 4.0 | never_ask | 85.00% | 8.3 | 0.00 | 1.94 | 98.00% |
| 3 | 0.0 | 4.0 | random_ask | 98.00% | 8.9 | 1.95 | 3.52 | 100.00% |
| 3 | 0.05 | 1.0 | always_ask | 87.00% | 9.7 | 2.28 | 4.19 | 98.00% |
| 3 | 0.05 | 1.0 | ask_or_act | 86.00% | 8.7 | 0.76 | 2.47 | 91.00% |
| 3 | 0.05 | 1.0 | info_gain_ask | 91.00% | 8.8 | 1.22 | 2.79 | 98.00% |
| 3 | 0.05 | 1.0 | never_ask | 76.00% | 9.2 | 0.00 | 2.59 | 83.00% |
| 3 | 0.05 | 1.0 | random_ask | 88.00% | 9.5 | 1.99 | 3.90 | 98.00% |
| 3 | 0.05 | 2.0 | always_ask | 88.00% | 10.2 | 2.36 | 4.34 | 98.00% |
| 3 | 0.05 | 2.0 | ask_or_act | 91.00% | 9.0 | 0.61 | 2.23 | 96.00% |
| 3 | 0.05 | 2.0 | info_gain_ask | 95.00% | 9.3 | 1.23 | 2.83 | 99.00% |
| 3 | 0.05 | 2.0 | never_ask | 84.00% | 9.2 | 0.00 | 2.20 | 96.00% |
| 3 | 0.05 | 2.0 | random_ask | 93.00% | 9.9 | 1.93 | 3.83 | 98.00% |
| 3 | 0.05 | 4.0 | always_ask | 91.00% | 9.2 | 2.18 | 3.89 | 98.00% |
| 3 | 0.05 | 4.0 | ask_or_act | 93.00% | 8.2 | 0.65 | 2.08 | 98.00% |
| 3 | 0.05 | 4.0 | info_gain_ask | 96.00% | 8.4 | 1.21 | 2.62 | 100.00% |
| 3 | 0.05 | 4.0 | never_ask | 91.00% | 8.2 | 0.00 | 1.75 | 98.00% |
| 3 | 0.05 | 4.0 | random_ask | 94.00% | 9.2 | 2.05 | 3.75 | 98.00% |
| 3 | 0.1 | 1.0 | always_ask | 91.00% | 9.2 | 2.12 | 3.85 | 95.00% |
| 3 | 0.1 | 1.0 | ask_or_act | 94.00% | 8.2 | 0.76 | 2.18 | 98.00% |
| 3 | 0.1 | 1.0 | info_gain_ask | 95.00% | 8.4 | 1.22 | 2.63 | 99.00% |
| 3 | 0.1 | 1.0 | never_ask | 78.00% | 8.7 | 0.00 | 2.34 | 94.00% |
| 3 | 0.1 | 1.0 | random_ask | 94.00% | 9.2 | 2.07 | 3.87 | 98.00% |
| 3 | 0.1 | 2.0 | always_ask | 94.00% | 9.5 | 2.34 | 4.11 | 98.00% |
| 3 | 0.1 | 2.0 | ask_or_act | 94.00% | 8.2 | 0.61 | 1.97 | 98.00% |
| 3 | 0.1 | 2.0 | info_gain_ask | 97.00% | 8.5 | 1.21 | 2.56 | 99.00% |
| 3 | 0.1 | 2.0 | never_ask | 88.00% | 8.4 | 0.00 | 1.84 | 95.00% |
| 3 | 0.1 | 2.0 | random_ask | 96.00% | 9.1 | 1.85 | 3.50 | 98.00% |
| 3 | 0.1 | 4.0 | always_ask | 91.00% | 9.3 | 2.26 | 4.02 | 98.00% |
| 3 | 0.1 | 4.0 | ask_or_act | 94.00% | 8.2 | 0.65 | 2.08 | 96.00% |
| 3 | 0.1 | 4.0 | info_gain_ask | 96.00% | 8.4 | 1.16 | 2.52 | 100.00% |
| 3 | 0.1 | 4.0 | never_ask | 85.00% | 8.6 | 0.00 | 2.15 | 95.00% |
| 3 | 0.1 | 4.0 | random_ask | 94.00% | 9.1 | 1.96 | 3.60 | 99.00% |
| 4 | 0.0 | 1.0 | always_ask | 83.00% | 10.0 | 2.50 | 4.47 | 95.00% |
| 4 | 0.0 | 1.0 | ask_or_act | 80.00% | 9.3 | 0.68 | 2.84 | 91.00% |
| 4 | 0.0 | 1.0 | info_gain_ask | 92.00% | 9.1 | 1.37 | 3.00 | 98.00% |
| 4 | 0.0 | 1.0 | never_ask | 66.00% | 9.6 | 0.00 | 2.78 | 92.00% |
| 4 | 0.0 | 1.0 | random_ask | 90.00% | 9.9 | 2.27 | 4.21 | 97.00% |
| 4 | 0.0 | 2.0 | always_ask | 92.00% | 10.1 | 2.62 | 4.53 | 98.00% |
| 4 | 0.0 | 2.0 | ask_or_act | 93.00% | 8.6 | 0.47 | 1.92 | 96.00% |
| 4 | 0.0 | 2.0 | info_gain_ask | 96.00% | 9.0 | 1.32 | 2.75 | 97.00% |
| 4 | 0.0 | 2.0 | never_ask | 86.00% | 9.0 | 0.00 | 2.08 | 97.00% |
| 4 | 0.0 | 2.0 | random_ask | 95.00% | 9.7 | 2.11 | 3.81 | 98.00% |
| 4 | 0.0 | 4.0 | always_ask | 94.00% | 9.9 | 2.52 | 4.43 | 100.00% |
| 4 | 0.0 | 4.0 | ask_or_act | 95.00% | 8.4 | 0.50 | 1.89 | 99.00% |
| 4 | 0.0 | 4.0 | info_gain_ask | 99.00% | 8.8 | 1.35 | 2.75 | 100.00% |
| 4 | 0.0 | 4.0 | never_ask | 90.00% | 8.5 | 0.00 | 1.81 | 99.00% |
| 4 | 0.0 | 4.0 | random_ask | 97.00% | 9.5 | 2.12 | 3.87 | 100.00% |
| 4 | 0.05 | 1.0 | always_ask | 84.00% | 10.1 | 2.59 | 4.63 | 98.00% |
| 4 | 0.05 | 1.0 | ask_or_act | 77.00% | 9.2 | 0.74 | 2.76 | 95.00% |
| 4 | 0.05 | 1.0 | info_gain_ask | 88.00% | 9.0 | 1.28 | 2.85 | 98.00% |
| 4 | 0.05 | 1.0 | never_ask | 68.00% | 9.6 | 0.00 | 2.78 | 85.00% |
| 4 | 0.05 | 1.0 | random_ask | 83.00% | 9.8 | 2.05 | 4.07 | 100.00% |
| 4 | 0.05 | 2.0 | always_ask | 89.00% | 10.1 | 2.51 | 4.46 | 98.00% |
| 4 | 0.05 | 2.0 | ask_or_act | 89.00% | 8.9 | 0.56 | 2.32 | 100.00% |
| 4 | 0.05 | 2.0 | info_gain_ask | 91.00% | 9.1 | 1.27 | 2.85 | 100.00% |
| 4 | 0.05 | 2.0 | never_ask | 83.00% | 9.1 | 0.00 | 2.23 | 95.00% |
| 4 | 0.05 | 2.0 | random_ask | 90.00% | 9.8 | 2.11 | 3.92 | 98.00% |
| 4 | 0.05 | 4.0 | always_ask | 90.00% | 9.8 | 2.62 | 4.62 | 100.00% |
| 4 | 0.05 | 4.0 | ask_or_act | 96.00% | 8.4 | 0.52 | 2.22 | 98.00% |
| 4 | 0.05 | 4.0 | info_gain_ask | 98.00% | 8.6 | 1.24 | 2.73 | 100.00% |
| 4 | 0.05 | 4.0 | never_ask | 95.00% | 8.4 | 0.00 | 1.98 | 98.00% |
| 4 | 0.05 | 4.0 | random_ask | 92.00% | 9.3 | 2.07 | 3.88 | 100.00% |
| 4 | 0.1 | 1.0 | always_ask | 77.00% | 9.5 | 2.64 | 4.74 | 96.00% |
| 4 | 0.1 | 1.0 | ask_or_act | 77.00% | 8.7 | 0.85 | 3.00 | 96.00% |
| 4 | 0.1 | 1.0 | info_gain_ask | 83.00% | 8.5 | 1.31 | 3.10 | 98.00% |
| 4 | 0.1 | 1.0 | never_ask | 60.00% | 9.3 | 0.00 | 3.21 | 88.00% |
| 4 | 0.1 | 1.0 | random_ask | 81.00% | 9.2 | 2.11 | 4.17 | 98.00% |
| 4 | 0.1 | 2.0 | always_ask | 85.00% | 9.5 | 2.56 | 4.53 | 100.00% |
| 4 | 0.1 | 2.0 | ask_or_act | 88.00% | 8.3 | 0.58 | 2.34 | 97.00% |
| 4 | 0.1 | 2.0 | info_gain_ask | 89.00% | 8.5 | 1.29 | 2.90 | 99.00% |
| 4 | 0.1 | 2.0 | never_ask | 76.00% | 8.7 | 0.00 | 2.45 | 93.00% |
| 4 | 0.1 | 2.0 | random_ask | 87.00% | 9.3 | 2.17 | 4.12 | 100.00% |
| 4 | 0.1 | 4.0 | always_ask | 84.00% | 9.6 | 2.62 | 4.64 | 97.00% |
| 4 | 0.1 | 4.0 | ask_or_act | 91.00% | 8.4 | 0.60 | 2.48 | 97.00% |
| 4 | 0.1 | 4.0 | info_gain_ask | 89.00% | 8.5 | 1.33 | 2.94 | 99.00% |
| 4 | 0.1 | 4.0 | never_ask | 86.00% | 8.6 | 0.00 | 2.32 | 97.00% |
| 4 | 0.1 | 4.0 | random_ask | 91.00% | 9.2 | 2.06 | 3.93 | 97.00% |

## Figures

### Main Dashboard

![Main dashboard](main_dashboard.png)

### Clarification Quality

![Clarification quality entropy delta](clarification_quality_entropy_delta.png)

### Ablations Dashboard

![Ablations dashboard](ablations_dashboard.png)

### Robustness: Answer Noise Deltas

![Robust answer noise deltas](robust_answer_noise_deltas.png)

### Robustness: Principal-Model Mismatch Deltas

![Robust mismatch deltas](robust_mismatch_deltas.png)


---

## Aggregate by K (averaged over eps and beta)

| K | Policy | Success rate | Avg steps | Avg questions | Avg regret | MAP correct |
|---|--------|--------------|-----------|---------------|------------|-------------|
| 1 | always_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | ask_or_act | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | info_gain_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | never_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 1 | random_ask | 100.00% | 6.8 | 0.00 | 0.53 | 100.00% |
| 2 | always_ask | 100.00% | 8.6 | 1.72 | 3.07 | 100.00% |
| 2 | ask_or_act | 91.11% | 8.1 | 0.00 | 1.67 | 96.56% |
| 2 | info_gain_ask | 100.00% | 7.9 | 1.00 | 2.00 | 100.00% |
| 2 | never_ask | 91.11% | 8.1 | 0.00 | 1.67 | 96.56% |
| 2 | random_ask | 99.89% | 8.6 | 1.72 | 3.07 | 100.00% |
| 3 | always_ask | 90.22% | 9.4 | 2.29 | 4.08 | 97.89% |
| 3 | ask_or_act | 91.67% | 8.3 | 0.67 | 2.14 | 96.78% |
| 3 | info_gain_ask | 94.78% | 8.5 | 1.21 | 2.61 | 99.22% |
| 3 | never_ask | 83.78% | 8.6 | 0.00 | 2.15 | 94.22% |
| 3 | random_ask | 93.22% | 9.2 | 1.96 | 3.67 | 98.78% |
| 4 | always_ask | 86.44% | 9.9 | 2.58 | 4.56 | 98.00% |
| 4 | ask_or_act | 87.33% | 8.7 | 0.61 | 2.42 | 96.56% |
| 4 | info_gain_ask | 91.67% | 8.8 | 1.31 | 2.88 | 98.78% |
| 4 | never_ask | 78.89% | 9.0 | 0.00 | 2.40 | 93.78% |
| 4 | random_ask | 89.56% | 9.5 | 2.12 | 4.00 | 98.67% |

---

## High-Ambiguity Snapshot (K=3,4)

- **always_ask:** success 88.33%, avg regret 4.32, avg questions 2.43.
- **ask_or_act:** success 89.50%, avg regret 2.28, avg questions 0.64.
- **info_gain_ask:** success 93.22%, avg regret 2.74, avg questions 1.26.
- **never_ask:** success 81.33%, avg regret 2.28, avg questions 0.00.
- **random_ask:** success 91.39%, avg regret 3.83, avg questions 2.04.

## Clarification Quality (First Asked Question)

We summarize posterior contraction from the first asked question using entropy and effective goal count.

| Policy | Ask rate (K=3,4) | H_before | H_after | DeltaH | N_eff_before | N_eff_after | DeltaN_eff | IG_first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_ask | 100.00% | 1.242 | 0.751 | 0.492 | 3.500 | 2.352 | 1.148 | 0.492 |
| ask_or_act | 63.22% | 1.209 | 0.755 | 0.454 | 3.354 | 2.358 | 0.996 | 0.460 |
| info_gain_ask | 100.00% | 1.242 | 0.497 | 0.745 | 3.500 | 1.734 | 1.766 | 0.746 |
| never_ask | 0.00% | 1.242 | 1.242 | 0.000 | 3.500 | 3.500 | 0.000 | 0.000 |
| random_ask | 100.00% | 1.242 | 0.877 | 0.366 | 3.500 | 2.629 | 0.871 | 0.361 |

Interpretation: `info_gain_ask` achieves the largest average first-question entropy reduction among policies that ask at K=3-4.

## Robustness Summary (K=3,4)

All results are averaged over replicate seeds; uncertainty is 95% bootstrap CI over episodes.

| K | Policy | Success (mean [95% CI]) | Regret (mean [95% CI]) | Questions (mean [95% CI]) |
|---|--------|--------------------------|-------------------------|----------------------------|
| 3 | always_ask | 90.22% [88.33%, 92.11%] | 4.08 [3.97, 4.19] | 2.29 [2.23, 2.35] |
| 3 | ask_or_act | 91.67% [89.78%, 93.44%] | 2.14 [2.05, 2.25] | 0.67 [0.63, 0.70] |
| 3 | info_gain_ask | 94.78% [93.33%, 96.22%] | 2.61 [2.54, 2.69] | 1.21 [1.19, 1.24] |
| 3 | never_ask | 83.78% [81.22%, 86.22%] | 2.15 [2.02, 2.27] | 0.00 [0.00, 0.00] |
| 3 | random_ask | 93.22% [91.66%, 94.67%] | 3.67 [3.58, 3.77] | 1.96 [1.91, 2.01] |
| 4 | always_ask | 86.44% [84.00%, 88.34%] | 4.56 [4.46, 4.66] | 2.58 [2.52, 2.62] |
| 4 | ask_or_act | 87.33% [85.11%, 89.34%] | 2.42 [2.30, 2.53] | 0.61 [0.58, 0.64] |
| 4 | info_gain_ask | 91.67% [89.67%, 93.33%] | 2.88 [2.79, 2.96] | 1.31 [1.27, 1.34] |
| 4 | never_ask | 78.89% [75.89%, 81.34%] | 2.40 [2.28, 2.53] | 0.00 [0.00, 0.00] |
| 4 | random_ask | 89.56% [87.44%, 91.33%] | 4.00 [3.90, 4.09] | 2.12 [2.06, 2.17] |

## Policy Difference CIs (K=3,4)

Deltas are bootstrap-estimated on paired episode keys: (K, eps, beta, rep_seed, episode_id).

| Contrast | Delta mean [95% CI] | N paired episodes |
|----------|-----------------------|-------------------|
| DeltaSuccess (AskOrAct - NeverAsk) | 8.17% [6.61%, 9.78%] | 1800 |
| DeltaRegret (AskOrAct - AlwaysAsk) | -2.04 [-2.13, -1.95] | 1800 |
| DeltaSuccess (AskOrAct - InfoGainAsk) | -3.72% [-4.94%, -2.61%] | 1800 |
| DeltaRegret (AskOrAct - InfoGainAsk) | -0.46 [-0.53, -0.40] | 1800 |

## Robustness Sweeps (K=3,4)

Robustness is summarized with paired deltas and bootstrap CIs under answer-noise shifts and principal-model mismatch.

### Answer-Noise Robustness

| answer_noise | DeltaSuccess (AskOrAct - NeverAsk) | DeltaRegret (AskOrAct - AlwaysAsk) |
|---:|---:|---:|
| 0.0 | 3.50% [-0.50%, 7.50%] | -2.14 [-2.42, -1.85] |
| 0.1 | 0.50% [-3.50%, 4.50%] | -2.60 [-2.84, -2.35] |
| 0.2 | 2.00% [-1.50%, 5.50%] | -2.98 [-3.21, -2.75] |

Trend: AskOrAct keeps a positive success advantage vs NeverAsk and a strong regret advantage vs AlwaysAsk as answer noise increases.

### Principal-Model Mismatch Robustness

| principal_beta | DeltaSuccess (AskOrAct - NeverAsk) | DeltaRegret (AskOrAct - AlwaysAsk) |
|---:|---:|---:|
| 1.0 | 13.50% [8.00%, 19.50%] | -1.69 [-1.98, -1.37] |
| 2.0 | 4.00% [0.50%, 7.50%] | -2.12 [-2.40, -1.86] |
| 4.0 | 6.00% [2.50%, 9.50%] | -2.37 [-2.63, -2.10] |

Trend: AskOrAct remains robust to principal rationality mismatch, preserving positive success deltas and negative regret deltas.

## Failure Mode Breakdown

| K | Policy | Failure rate | failure_by_timeout rate | failure_by_wrong_pick rate |
|---|--------|--------------|-------------------------|----------------------------|
| 3 | always_ask | 9.78% | 9.78% | 0.00% |
| 3 | ask_or_act | 8.33% | 8.33% | 0.00% |
| 3 | info_gain_ask | 5.22% | 5.22% | 0.00% |
| 3 | never_ask | 16.22% | 16.22% | 0.00% |
| 3 | random_ask | 6.78% | 6.78% | 0.00% |
| 4 | always_ask | 13.56% | 13.56% | 0.00% |
| 4 | ask_or_act | 12.67% | 12.67% | 0.00% |
| 4 | info_gain_ask | 8.33% | 8.33% | 0.00% |
| 4 | never_ask | 21.11% | 21.11% | 0.00% |
| 4 | random_ask | 10.44% | 10.44% | 0.00% |

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
  info_gain_ask -> success 92.36%, regret 2.03, questions 1.23
  random_ask -> success 86.11%, regret 2.58, questions 1.98
- **modeB (K=3,4, wrong_pick_fail=False):**
  ask_or_act -> success 90.97%, regret 1.66, questions 0.65
  never_ask -> success 82.64%, regret 2.22, questions 0.00
  always_ask -> success 95.83%, regret 2.03, questions 2.44
  info_gain_ask -> success 92.36%, regret 1.43, questions 1.24
  random_ask -> success 96.53%, regret 1.90, questions 2.03
- **modeC (K=3,4, wrong_pick_fail=False):**
  ask_or_act -> success 87.50%, regret 2.37, questions 0.67
  never_ask -> success 84.03%, regret 1.98, questions 0.00
  always_ask -> success 78.47%, regret 4.36, questions 2.51
  info_gain_ask -> success 93.75%, regret 2.66, questions 1.23
  random_ask -> success 90.28%, regret 3.61, questions 1.97

---

*Raw data: `results/metrics.csv`*
*Main plots: `results/main_dashboard.png`*
*Clarification quality plot: `results/clarification_quality_entropy_delta.png`*
*Ablations: `results/metrics_ablations.csv` and `results/ablations_dashboard.png`*
*Robustness plots: `results/robust_answer_noise_deltas.png`, `results/robust_mismatch_deltas.png`*

## Robustness Sweep Artifacts

- `results/metrics_robust_answer_noise.csv`
- `results/metrics_robust_mismatch.csv`
- `results/robust_answer_noise_deltas.png`
- `results/robust_mismatch_deltas.png`
