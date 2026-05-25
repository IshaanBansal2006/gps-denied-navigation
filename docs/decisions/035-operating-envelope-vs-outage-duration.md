# Decision 035: Operating Envelope — Which System Wins at Which Outage Duration?

**Date:** 2026-05-24 (scaffold; numbers to be filled in when eval completes)
**Status:** Pending — `scripts/multi_duration_eval.py` waiting for free desktop
GPU (queued after cross_seq + monte_carlo).

## Context

Every prior decision (017–034) reported results at a *single* outage
duration, usually 30 s — the headline scenario. But the real operating
envelope of a GPS-denied navigation system is the full curve: how does
each system degrade as the outage gets longer? Where does the RLS
adapter's setup overhead start to pay off? Where do all systems fail?

Decision 029 already noted that RLS hurts on 5 / 10 / 60 s outages while
helping on 30 s. This decision quantifies that observation across a full
sweep and produces the duration-curve figure for the README.

## Approach

`scripts/multi_duration_eval.py` runs three systems × seven outage
durations from one checkpoint on MH_05:

- **Systems:** vanilla v15 + filter, v15 + RLS, v15 + continuous.
- **Durations:** 5, 10, 15, 20, 30, 45, 60 s.
- **Outage start fraction:** 0.4 (matches the published headline).

`scripts/make_duration_curve_figure.py` renders a 2-panel side-by-side
plot: final velocity error vs duration, final position drift vs duration.

If `lstm_v18.pt` is available, the same sweep runs for v18 too and overlays
both checkpoints in dashed style.

## Results — TO BE FILLED IN

### v15 checkpoint

| Duration | Vanilla v15 | + RLS | + Continuous | Winner |
|---|---:|---:|---:|---|
| 5 s   | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 s  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 15 s  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 20 s  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 30 s  | 0.403 | **0.259** | 0.246 | _TBD_ |
| 45 s  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 60 s  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

(Final velocity error at outage end, m/s. Reference 30 s numbers from
decisions 029 and 032.)

### v18 checkpoint (if v18 finishes in time)

_TBD_

## Findings — TO BE FILLED IN

### Where does RLS start winning?
_TBD: hypothesis is around 20-25 s, since RLS specialization needs time
to integrate the per-step error reduction into a final-position benefit._

### Where does everything fail?
_TBD: at 60 s the v15 IMU drift is huge regardless of model. Expect all
three to be ~similar at 60 s, gap to oracle dominant._

### Does continuous adaptation help across the curve or only at 30 s?
_TBD: decision 032 had a 30 s test win that didn't replicate on the val
sequence. If continuous wins broadly across durations on the test
sequence, that strengthens the case; if it only wins at 30 s, that's a
red flag for the single-point methodology._

## Decision — TO BE FILLED IN

_TBD: based on the curve, recommend which system to deploy at which
duration. The README's ablation table currently has a fixed-30s view;
this decision proposes either replacing it with the duration curve or
adding the curve as a complement._

## What this changes about the project

- New eval script: `scripts/multi_duration_eval.py`.
- New figure: `docs/figures/duration_curve.{png,svg}`.
- New figure script: `scripts/make_duration_curve_figure.py`.
- README "Headline result" section gets the duration curve as a
  complement to the bar chart.
- README "What's honest about this" section adds an explicit operating-
  envelope statement (vs the current "only helps at 30 s" caveat).
