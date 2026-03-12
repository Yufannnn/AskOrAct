# Ask or Act for Cooperative Assistance via Inverse Planning
## CS6208 Final Presentation — Zhu Yufan (A0238993J)

---

## Slide 1 — Title
**Ask or Act for Cooperative Assistance via Inverse Planning**
CS6208 Final Presentation · April 2026
Zhu Yufan · A0238993J

---

## Slide 2 — Problem
- A helper must decide: **act now** or **ask a clarifying question**?
- Under an ambiguous instruction and a hidden goal, acting too early risks the wrong object.
- Asking too often wastes steps and is annoying.
- **Goal**: find the policy that asks *only when it helps*.

---

## Slide 3 — Setup
- 9×9 two-room gridworld, 6 objects, BFS shortest paths.
- Principal has a **hidden goal object**; gives an ambiguous instruction.
- Instruction maps to **K candidate goals** (ambiguity level K ∈ {1,2,3,4}).
- Principal acts approximately rationally (Boltzmann, β=rationality, ε=noise).
- No LLMs: templated instructions, fixed 3-question menu, fixed answer model.

---

## Slide 4 — Method: Bayesian Posterior
- Initialise posterior b(g) from instruction likelihood over K candidates.
- After each principal action a at state s:
  - b(g) ← b(g) · π(a | s, g) [inverse planning update]
- After asking question q and receiving answer α:
  - b(g) ← b(g) · P(α | q, g) [answer likelihood update]

---

## Slide 5 — Method: Ask-or-Act Decision
At each step compare:
- **CostAct** = expected path length under current posterior
- **CostAsk(q)** = 1 + 0.5 (question cost) + E[CostAct after update]

Ask argmin_q CostAsk(q) if it beats CostAct, subject to:
1. Entropy gate: skip if H(b) ≤ 0.3 (already confident)
2. Ask window: stop asking after step 6
3. Budget: max 3 questions per episode

---

## Slide 6 — Policies Compared (7 total)
| Policy | Strategy |
|--------|----------|
| **AskOrAct** | Ask only when VoI exceeds cost |
| NeverAsk | Always act on MAP goal |
| AlwaysAsk | Ask until budget exhausted |
| InfoGainAsk | Ask highest-IG question when uncertain |
| EasyInfoGainAsk | InfoGainAsk, max 1 question |
| RandomAsk | Same gate as InfoGain, random question |
| POMCPPlanner | POMDP tree search baseline |

---

## Slide 7 — Evaluation Protocol
- K ∈ {1,2,3,4} × ε ∈ {0,0.05,0.1} × β ∈ {1,2,4}
- 5 replicate seeds × 20 episodes = **100 episodes per cell**
- **Total: 19,800 episodes** across all policies
- Metrics: success rate, regret (vs oracle), questions/ep, MAP accuracy
- Uncertainty: **95% bootstrap CIs** (B=1000 resamples)

---

## Slide 8 — Main Results *(show results/main_dashboard.png)*
- K=1: all policies identical (100% success, 0 questions) — good sanity check
- K=2: AskOrAct **96.7%** vs NeverAsk 91.1% (+5.6 pp), asks 0.65 q/ep
- K=3: AskOrAct **91.7%** vs NeverAsk 83.8% (**+7.9 pp**, CI [5.0, 10.8])
- K=4: AskOrAct **87.3%** vs NeverAsk 78.9% (**+8.4 pp**, CI [5.1, 12.1])
- Both K=3,4 CIs **exclude zero** → statistically significant

---

## Slide 9 — Cost of Over-Asking and Under-Asking
| Policy | K=4 Success | K=4 Regret | Q/ep |
|--------|------------|------------|------|
| InfoGainAsk | **91.7%** | 2.88 | 1.31 |
| **AskOrAct** | 87.3% | **2.42** | 0.61 |
| AlwaysAsk | 86.4% | 4.56 | 2.58 |
| NeverAsk | 78.9% | 2.40 | 0.00 |

- AlwaysAsk asks 4× more than AskOrAct but achieves lower success and **+2.1 higher regret**
- AskOrAct achieves the best **regret** at K=3 (2.14) and near-best at K=4

---

## Slide 10 — Pareto Frontier *(show results/pareto_K4.png)*
- Plot: success rate vs regret at K=4
- **AskOrAct is Pareto-efficient**: dominates NeverAsk in success, dominates AlwaysAsk/RandomAsk in regret
- InfoGainAsk gets higher success but at a regret cost

---

## Slide 11 — Robustness
**Answer noise** (swept 0 → 0.2, K=4):
- AskOrAct advantage holds at all noise levels (+4 to +9 pp)

**Principal model mismatch** (β̂=2, true β∈{1,2,4}):
- Hardest: true β=1 (near-random) → AskOrAct 83.5% vs NeverAsk 70.0% (+13.5 pp)
- Matched β=2 → 92.5% vs 88.5% (+4 pp)

**Held-out templates + scale-K (K=5,6)**:
- Policy ordering preserved; asking advantage grows with K

---

## Slide 12 — Ablations
| Mode | Question cost | Ask=step? | Success | Q/ep |
|------|--------------|-----------|---------|------|
| A | 0.0 | yes | 69.2% | 1.61 |
| B | 0.5 | no  | 73.6% | 0.64 |
| **C** | **0.5** | **yes** | **73.8%** | **0.65** |

- **Mode A (cost=0)**: asks 2.5× more, achieves *lower* success → question cost is essential
- Mode B vs C: ask-counts-as-step has minimal effect once cost is non-zero

---

## Slide 13 — Takeaways
1. **Selective asking works**: AskOrAct improves success by ~8 pp at K=3,4 with <1 question/ep
2. **Both extremes fail**: NeverAsk under-queries; AlwaysAsk over-queries and pays high regret
3. **Question cost matters**: zero-cost asking leads to counter-productive over-asking
4. **Robust**: advantage survives answer noise, model mismatch, and held-out templates

---

## Slide 14 — Limitations & Future Work
**Current limits:**
- Small gridworld; exact posterior enumeration
- Fixed 3-question menu; templated language only
- Simulated Boltzmann principal

**Future directions:**
- Larger environments with approximate inference (particle filter)
- Open-ended natural language questions
- Learning question cost from interaction
- Real human principals

---

## Backup — Bug Found & Fixed During Evaluation
- **Bug**: `ENTROPY_GATE = 0.8` caused AskOrAct to never ask at K=2
  - Uniform prior over K=2 goals has entropy ln(2) ≈ 0.69 < 0.8 → gate always fires
  - Pre-fix K=2 results: AskOrAct identical to NeverAsk (0 questions asked)
- **Fix**: reduced `ENTROPY_GATE` to 0.3
- Post-fix K=2: AskOrAct 96.7% vs NeverAsk 91.1% (+5.6 pp)
- *Finding and correcting evaluation bugs is part of rigorous empirical research*
