# Executive Summary

This project studies cooperative assistance under ambiguity: should a helper ask a clarifying question or act immediately?

We implement a controlled gridworld with:
- hidden principal goal,
- ambiguous templated instruction,
- principal behavior generated from an approximately rational policy,
- assistant policies: NeverAsk, AlwaysAsk, AskOrAct.

AskOrAct maintains a posterior over goals from instruction, observed principal actions, and optional answers. It compares expected team cost of acting now versus asking a question and chooses the lower-cost option.

Main evaluation includes ambiguity/noise/rationality sweeps, replicate-seed robustness, and 95% bootstrap confidence intervals. We report success, regret, questions asked, MAP correctness, and failure modes.

Key high-ambiguity result (K=3,4):
- DeltaSuccess (AskOrAct - NeverAsk): positive with non-overlapping CI from zero in this run.
- DeltaRegret (AskOrAct - AlwaysAsk): strongly negative with tight CI.

Interpretation:
- NeverAsk under-queries and commits wrong more often under ambiguity.
- AlwaysAsk over-queries and pays unnecessary cost.
- AskOrAct learns a middle policy: ask when uncertainty reduction is worth the cost.

The codebase is reproducible and packaged for submission via:
- `./run.sh all`
- `./run.sh package`
