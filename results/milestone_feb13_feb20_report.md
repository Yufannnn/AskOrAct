# Milestone Feb 13 + Feb 20 — Results Report

**Generated:** 2026-02-18 00:23

## Setup

- **Feb 13:** Gridworld tasks, goal sets, ambiguous instruction templates, scripted (approximately rational) principal.
- **Feb 20:** Posterior inference over goals from instruction + observed principal actions; act-only assistant (no questions).
- **Policy:** Assistant always acts toward current MAP goal; posterior updated each step from principal action.
- **Conditions:** Ambiguity K ∈ {1, 2, 3, 4} (number of candidate goals matching instruction), principal noise eps ∈ {0.0, 0.05, 0.1}.
- **Episodes per condition:** 15.

---

## Summary by condition

| Ambiguity K | eps | Success rate | Avg steps | MAP correct (final) | N episodes |
|-------------|-----|--------------|-----------|----------------------|------------|
| 1 | 0.0 | 100.00% | 12.1 | 100.00% | 15 |
| 1 | 0.05 | 100.00% | 13.7 | 100.00% | 15 |
| 1 | 0.1 | 100.00% | 15.3 | 100.00% | 15 |
| 2 | 0.0 | 100.00% | 8.9 | 100.00% | 15 |
| 2 | 0.05 | 100.00% | 8.5 | 100.00% | 15 |
| 2 | 0.1 | 100.00% | 8.7 | 100.00% | 15 |
| 3 | 0.0 | 100.00% | 8.3 | 100.00% | 15 |
| 3 | 0.05 | 100.00% | 10.9 | 100.00% | 15 |
| 3 | 0.1 | 100.00% | 13.5 | 100.00% | 15 |
| 4 | 0.0 | 100.00% | 9.9 | 93.33% | 15 |
| 4 | 0.05 | 100.00% | 10.5 | 93.33% | 15 |
| 4 | 0.1 | 100.00% | 12.9 | 100.00% | 15 |

---

## Aggregate (all conditions)

- **Overall success rate:** 100.00% (180/180)
- **Overall average steps (when run to end):** 11.1
- **Final MAP goal correct (when run ended):** 98.89%

---

*Metrics and raw rows are in `milestone_feb13_feb20_metrics.csv` and `milestone_feb13_feb20_summary.json`.*
