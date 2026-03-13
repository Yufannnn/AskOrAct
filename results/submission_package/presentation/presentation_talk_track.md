# Presentation Talk Track

## 1-minute opening
- We study a cooperative assistant that decides whether to ask a clarifying question or act immediately.
- The setting is a controlled gridworld with ambiguous instructions and hidden goals.
- We compare three policies: NeverAsk, AlwaysAsk, and AskOrAct.
- Main result: at high ambiguity, AskOrAct improves success over NeverAsk and achieves much lower regret than AlwaysAsk.

## 5-minute flow
1. Problem and motivation
   - Clarification has value, but also cost.
   - We need a principled ask-vs-act decision.
2. Model
   - Posterior over goals from instruction plus principal behavior.
   - Expected team-cost comparison between acting now and asking best question.
3. Experimental setup
   - Ambiguity, principal noise, rationality parameters.
   - Deterministic seeds and replicate runs.
4. Results
   - Show main dashboard: success, regret, questions, MAP with CIs.
   - Highlight high-ambiguity deltas:
     - DeltaSuccess (AskOrAct - NeverAsk): positive.
     - DeltaRegret (AskOrAct - AlwaysAsk): negative.
5. Conclusion
   - Selective asking is the best tradeoff in this controlled setting.

## 10-minute flow (add details)
1. Environment details
   - Assistant-only completion semantics.
   - Dynamic deadline and ask time cost.
   - Candidate goals separated to make ambiguity meaningful.
2. AskOrAct safeguards
   - entropy gate
   - ask window
   - max question budget
   - no repeated questions in an episode
3. Failure semantics and ablations
   - timeout-driven failures in main sweep
   - optional wrong-pick fail for harder setting
   - Mode A/B/C to separate time-cost and communication-cost effects
4. Robustness
   - replicate seeds and 95% bootstrap CIs
   - paired delta CI for policy differences

## Likely questions and short answers
- Why not AlwaysAsk?
  - Asking has cost; AlwaysAsk overpays under low uncertainty and can hurt under deadlines.
- Why not NeverAsk?
  - Under high ambiguity it commits wrong too often, reducing success.
- Is this just tuned heuristics?
  - The decision rule is expected-cost based with probabilistic inference; gates control over-asking.
- How robust are results?
  - We report replicate-seed averages and 95% bootstrap CIs, including paired delta CIs.
- Does this generalize beyond templates?
  - The current claim is for controlled template language; extension to richer language is future work.

## Demo plan (if asked)
- Show one AskOrAct debug episode at `K=4`.
- Highlight early question then commitment.
- Point to `./run.sh quick-demo` and `./run.sh report` for reproducibility.
