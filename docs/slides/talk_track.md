# Presentation Talk Track — 5-Minute Word-by-Word Script

Target pace: ~150 words per minute, ~550 words of script + ~80s of
pauses and transitions ≈ 5 minutes total. Pauses marked with [pause].

## Slide 1 — Title (~25 seconds)

Good afternoon. I'm Yufan, and my project is called
"When More Observation Hurts.
Redundancy and Robust Clarification in Ask-or-Act" [pause] Imagine an assistant trying
to help someone, but it's not sure what they want. It has two ways
to find out — it can watch what they do, or it can stop and ask.
My project looks at how these two options interact, and it turns
out that sometimes watching more actually makes the assistant
worse, not better.

## Slide 2 — The Setup (~75 seconds)

Here's the setup. A principal and an assistant share a small
gridworld. The principal has a goal in mind; the assistant doesn't
know which one. Every step, the assistant can either act — and
hope it guessed right — or spend a little time asking a
clarifying question. [pause]

I compare three simple policies. **NeverAsk** never spends the
question cost: it only watches the principal, updates its belief,
and then commits. **AlwaysAsk** does the opposite: as long as it's
still unsure, it keeps asking for clarification before acting.
And **AskOrAct** is the main policy: at each step, it compares the
expected cost of moving now against the expected cost of asking one
question first, and picks the cheaper option. [pause]

## Slide 3 — Two Surprises (~95 seconds)

I want to show you two findings that surprised me. [pause]

First one, on the left. You'd think watching and asking are two
independent sources of information, so their benefits should add
up. They don't. They overlap — a lot. The practical lesson is
that asking is most valuable exactly when watching fails you.
When the principal's first move is ambiguous, one good question
is worth more than any amount of further observation. [pause]

Second one, on the right, is weirder. You'd also think a more
rational principal is easier to help. After all, a rational
principal takes sensible actions. But as the principal gets more
rational, they stop varying — they just take the single best
move, over and over. And that kills the assistant's ability to
tell the goals apart. So assistability goes up, and then it comes
back down. A smart assistant has to notice this and ask more
questions exactly when the principal looks most confident.

## Slide 4 — When Watching Hurts, and the Fix (~100 seconds)

Now the main finding. In the real world, the assistant's model of
the principal is never perfect. When the two don't match, every
extra step of passive watching doesn't just stop helping — it
actively makes things worse. The posterior drifts toward the
wrong goal, and the more you watch, the more confident you
become in the wrong answer. [pause]

My first instinct was to fix this at the decision layer — ask
sooner, ask more, tune the gates. None of that worked. The
bottleneck isn't when to ask. It's that the assistant keeps
trusting observations it shouldn't trust. [pause]

So the fix is simple. When an observation looks surprising under
the assistant's own model, we turn down how much we believe it.
When the model is matched, this does no harm. When the model is
wrong, it recovers most of the loss. [pause]

Three things to take away. One — watching and asking overlap,
so don't stack them, use them where each one earns its keep.
Two — a more rational partner is not always easier to help.
And three — when your model of the world is wrong, fix what
you believe, not when you speak. Thank you.

## Likely questions and short answers
- Why not AlwaysAsk? Asking has a cost; it overpays when
  uncertainty is low.
- Why not NeverAsk? Under high ambiguity it commits to the wrong
  goal too often.
- Is this just tuned heuristics? No — the decision rule is
  expected-cost based with a probabilistic posterior.
- How robust are the results? Replicate-seed averages with 95%
  bootstrap CIs and paired delta CIs.
- Does it generalize beyond templates? The claim is for the
  controlled setting; richer language is future work.
