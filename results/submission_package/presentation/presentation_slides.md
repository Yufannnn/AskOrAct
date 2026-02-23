# Ask or Act for Cooperative Assistance via Inverse Planning

## Slide 1 - Title
- Ask or Act for Cooperative Assistance via Inverse Planning
- CS6208 Project
- Team member: Zhu Yufan (A0238993J)

## Slide 2 - Problem
- A helper should decide whether to act now or ask a clarifying question.
- Under ambiguous instruction and hidden goal, acting too early can waste steps.
- Asking too much can also hurt (time cost and/or communication cost).

## Slide 3 - Environment and Assumptions
- Small cooperative gridworld with principal + assistant.
- Principal has hidden goal object.
- Instruction maps to K candidate goals (ambiguity level K).
- Principal behavior is approximately rational with noise.
- No LLMs: templated instructions, fixed question menu, fixed answer model.

## Slide 4 - Method
- Build posterior over candidate goals from:
  - instruction likelihood
  - observed principal actions (inverse planning update)
  - question answers (when asked)
- AskOrAct chooses lower expected team cost:
  - `CostAct`: act now toward MAP/expected value
  - `CostAsk`: ask best available question, then act with updated posterior
- Includes entropy gate, ask window, and max-question budget.

## Slide 5 - Policies Compared
- `NeverAsk`: always act.
- `AlwaysAsk`: always ask best question (up to limits).
- `AskOrAct`: ask only when expected cost improves.

## Slide 6 - Evaluation Protocol
- Sweep over:
  - `K in {1,2,3,4}`
  - `eps in {0.0, 0.05, 0.1}`
  - `beta in {1.0, 2.0, 4.0}`
- Replicates:
  - `REPL_SEEDS = [0,1,2,3,4]`
  - deterministic per-episode seed formula in code
- Metrics:
  - success, regret, questions asked, MAP accuracy
  - failure breakdown: timeout vs wrong-pick fail mode

## Slide 7 - Main Results (show `results/main_dashboard.png`)
- AskOrAct asks near-zero at low ambiguity and more at high ambiguity.
- At `K=3,4`, AskOrAct outperforms NeverAsk on success.
- AskOrAct has substantially lower regret than AlwaysAsk at high ambiguity.
- 95% bootstrap CIs are plotted for each policy curve.

## Slide 8 - Clear Statistical Separation
- Delta CI summary at `K in {3,4}`:
  - `DeltaSuccess (AskOrAct - NeverAsk) = 8.17% [6.61%, 9.78%]`
  - `DeltaRegret (AskOrAct - AlwaysAsk) = -2.04 [-2.13, -1.95]`
- Interpretation:
  - AskOrAct improves completion under ambiguity.
  - AskOrAct avoids over-questioning cost of AlwaysAsk.

## Slide 9 - Ablations (show `results/ablations_dashboard.png`)
- Mode A (time-only): ask consumes a step, no question fee.
- Mode B (comm-only): no step cost for ask, question fee only.
- Mode C (both): ask costs a step and a fee.
- Optional wrong-pick-fail switch increases hardness and sharpens policy separation.

## Slide 10 - Takeaways and Limits
- Core claim supported: selective asking beats always asking or never asking under ambiguity.
- Robustness included: seed replicates + bootstrap CIs + failure-mode reporting.
- Current limits:
  - templated language only
  - small gridworld
  - fixed question menu
- Next extension:
  - larger worlds, richer question design, stronger principal behavior models.

## Backup Slides (optional)
- Failure mode table (timeout vs wrong-pick).
- Seed robustness details.
- Hyperparameter sensitivity for entropy gate, ask window, and question budget.
