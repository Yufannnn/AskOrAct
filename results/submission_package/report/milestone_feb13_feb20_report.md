# Milestone Feb 13 + Feb 20 - Results Report

**Generated:** 2026-02-23 18:01

## Setup

- **Feb 13:** Gridworld tasks, goal sets, ambiguous instruction templates, scripted (approximately rational) principal.
- **Feb 20:** Posterior inference over goals from instruction + observed principal actions; act-only assistant (no questions).
- **Policy:** Assistant always acts toward current MAP goal; posterior updated each step from principal action.
- **Conditions:** Ambiguity K in {1, 2, 3, 4} (number of candidate goals matching instruction), principal noise eps in {0.0, 0.05, 0.1}.
- **Episodes per condition:** 15.

---

## Summary by condition

| Ambiguity K | eps | Success rate | Avg steps | MAP correct (final) | N episodes |
|-------------|-----|--------------|-----------|----------------------|------------|
| 1 | 0.0 | 100.00% | 6.9 | 100.00% | 15 |
| 1 | 0.05 | 100.00% | 7.4 | 100.00% | 15 |
| 1 | 0.1 | 100.00% | 7.6 | 100.00% | 15 |
| 2 | 0.0 | 100.00% | 7.6 | 100.00% | 15 |
| 2 | 0.05 | 100.00% | 6.8 | 100.00% | 15 |
| 2 | 0.1 | 100.00% | 7.3 | 93.33% | 15 |
| 3 | 0.0 | 100.00% | 10.3 | 100.00% | 15 |
| 3 | 0.05 | 100.00% | 9.9 | 100.00% | 15 |
| 3 | 0.1 | 100.00% | 9.6 | 86.67% | 15 |
| 4 | 0.0 | 100.00% | 10.4 | 100.00% | 15 |
| 4 | 0.05 | 100.00% | 9.3 | 100.00% | 15 |
| 4 | 0.1 | 100.00% | 9.3 | 100.00% | 15 |

---

## Aggregate (all conditions)

- **Overall success rate:** 100.00% (180/180)
- **Overall average steps (when run to end):** 8.5
- **Final MAP goal correct (when run ended):** 98.33%

---

*Metrics and raw rows are in `milestone_feb13_feb20_metrics.csv` and `milestone_feb13_feb20_summary.json`.*
