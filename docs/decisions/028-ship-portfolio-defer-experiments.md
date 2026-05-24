# Decision 028: Ship the Portfolio, Defer Further Experiments

**Date:** 2026-05-22
**Status:** Accepted

## Context

After v15 landed (decision 027) the open experimental directions were:

1. **Sequence-level adaptation** — TTT / LoRA / RLS-head that uses pre-outage
   GPS-aided velocity to specialize the model to the current sequence at inference
   time. Highest upside on the residual gap to the GPS oracle (0.405 → 0.104 m/s);
   highest implementation risk.
2. **v17: v15 retrained with val_final checkpoint selection** — one-line change,
   cheap, plausible 5–10% improvement based on the training-dynamic observation in
   decision 027.
3. **Longer-outage curriculum / bigger model** — moderate cost, unclear payoff.

Simultaneously, the Summer 2027 internship cycle opens in ~3 months and the
portfolio piece has zero current visibility: no compelling hero figure, no
README hook, no demo, no writeup, no distribution. The project has 26 decision
docs and 16 model variants but nothing a recruiter spends >30 seconds on.

## Decision

**Stop experimenting. Ship the portfolio.**

Pivoting all remaining bandwidth from modeling to packaging:
- One headline experiment (RLS adaptation head, the cheapest variant of #1) gets
  built and evaluated; if it lands a win, it goes in the README; if not, the
  attempt itself becomes a decision doc and the deferred-experiments story.
- Everything else from the experiment queue (#2, #3, and the heavier variants of
  #1) is deferred to post-recruiting cycle.

## Why

The marginal recruiter signal from a 0.405 → 0.38 m/s improvement is zero.
The marginal recruiter signal from a 30-second video of the model navigating
through a GPS outage is large. The lever now is *visibility*, not modeling.

Specific factors:
- The project already crosses the "real result" bar — v15 + filter outperforms
  the GPS oracle at the chosen hero-figure outage window (3.13 m vs 6.51 m final
  position drift on MH_05).
- The 4× gap to the GPS oracle on average is structural distribution shift, not
  solvable by per-step accuracy improvements — decision 022 established this and
  v15 confirmed it.
- Sequence-level adaptation has high research uncertainty. If it doesn't work it
  costs a week with nothing to show; if it works it might shift the headline
  number by 30%. Net expected value is positive but variance is high, and we are
  variance-averse with 3 days to ship.
- The 1-day RLS adaptation is the cheapest variant of #1 — it tests the hypothesis
  cheaply and either gets us a win or generates a defensible "we tried it" story.

## Plan

See `docs/plan-portfolio-ship.md` for the compressed 4-day timeline.

| Day | Output |
|---|---|
| 1 (Fri 5-22) | Decision docs 027 + 028, hero figure, loss-curve figure, baseline-comparison figure |
| 2 (Sat 5-23) | RLS adaptation head: implementation, training, eval, decision 029 |
| 3 (Sun 5-24) | Animated GIF for README, plot polish, repo cleanup |
| 4 (Mon 5-25) | README rewrite, Notion writeup, demo video/GIF embedded, LinkedIn announcement |
| (Tue 5-26) | Apply |

## Override note

Per `~/projects/CLAUDE.md` §1, AI code authorship is normally restricted to
scaffolding for these projects. The user has explicitly invoked the §6 override
for this project only, for the duration of this ship sprint — Claude is writing
the model and training code for the RLS experiment, the figure scripts, the
animations, and the README. The override does not extend to future projects.

## What's left on the table (defer queue)

After the ship, in order of expected value if the user returns to this project:

1. **Sequence-level adaptation, fuller form** — proper TTT or LoRA head, not just
   the cheapest linear variant. If RLS shows any signal, this is the obvious next
   step.
2. **v17 — v15 with val_final checkpoint selection** — quick consolation prize.
3. **Cross-dataset evaluation** — train on EuRoC, eval on TUM-VI or KITTI. Big
   recruiter signal but expensive to wire up.
4. **Longer-outage curriculum** — 60s, 120s outages with a curriculum schedule.

Each of these would be a future decision doc.
