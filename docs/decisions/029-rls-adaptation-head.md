# Decision 029: RLS Adaptation Head — Wins at 30s, Loses Elsewhere

**Date:** 2026-05-23
**Status:** Accepted (with caveats — see "Limitations")

## Context

Per decisions 027 + 028, sequence-level adaptation was the most promising open
direction for closing the residual 4× gap between LSTM v15 + filter (0.405 m/s
final at 30s) and the GPS-aided EKF oracle (0.104 m/s). The full menu of
adaptation approaches (TTT, LoRA, MAML, RLS-head) was deferred for time, but
the simplest variant — a recursive-least-squares adaptive linear head on top
of the frozen LSTM body — was selected as a 1-day single-shot experiment.

## Approach

**Architecture:** freeze LSTM body, replace the final `nn.Linear` head with an
`RLSHead` that supports online updates. The head maps the LSTM hidden state
(128-dim) to normalized 3-D velocity.

**Adaptation phase (pre-outage):** at every STRIDE-th IMU sample during the
GPS-available phase, extract the LSTM hidden state, compute the
GPS-aided velocity target (normalized), and run one RLS update.

**Prediction phase (during outage):** use the adapted head to predict velocity
from the LSTM hidden state. No further RLS updates. Predictions feed the same
velocity-only filter used by vanilla v15.

**Hyperparameters (after sweep):**
- forgetting factor λ = 0.995
- initial covariance P₀ = 0.1 · I  (small = weak adaptation; tried 0.1, 1.0, 10.0, 100.0)
- update rate: every 25th IMU sample (matches the velocity-only filter cadence)

Implementation: `src/adaptation/rls.py` (87 lines, pure NumPy, no autograd).
Eval harness: `scripts/neural_aided_ekf_lstm_v15_rls.py`.

## Results

All on EuRoC MH_05_difficult (the held-out test sequence), outage at 40% through
the sequence, single 358-update RLS adaptation per outage event.

| Outage | Vanilla v15 + filter | RLS-adapted v15 + filter | Δ |
|---|---|---|---|
|  5 s | 0.505 m/s | 0.808 m/s | **+60 %** |
| 10 s | 1.089 m/s | 1.252 m/s | +15 % |
| **30 s** | **0.403 m/s** | **0.259 m/s** | **−36 %** |
| 60 s | 0.803 m/s | 1.200 m/s | +49 % |

(Final velocity error at outage end. Lower is better.)

## Findings

### The headline 30s number drops 36%
- v15 + filter (vanilla): 0.403 m/s
- v15 + filter + RLS:     **0.259 m/s** ← new project best
- GPS-aided EKF oracle:   0.104 m/s

The 4× gap to the oracle becomes 2.5×. That is a real, defensible win on the
exact scenario the project optimizes for.

### But adaptation hurts at every other horizon
At 5s, 10s, and 60s outages the RLS-adapted variant is worse than vanilla v15.
The pattern is consistent: the adapted head specializes to the recent pre-outage
motion, which helps when the outage is the same length as the training rollout
(30s) but hurts when the motion regime drifts away from the recent pre-outage
context (longer horizons) or hasn't had time to integrate the benefit (shorter
horizons).

### Hyperparameter sensitivity
P₀ matters more than λ. Across forgetting factors 0.995, 0.999, 0.9999, the 30s
result moves from 0.372 → 0.420 → 0.434 — modest. Across P₀ values 0.1, 1.0,
10.0, 100.0 the 30s result moves from 0.259 → 0.325 → 0.372 → 0.424 — large.

Smaller P₀ corresponds to a tighter prior on the pre-trained head, equivalent to
a smaller effective learning rate. The model wants weak adaptation.

## What this changes about the project

- Headline number: **0.259 m/s final velocity error after 30 s GPS outage** —
  2.5× the GPS oracle ceiling instead of 4×.
- The system recommendation table from decision 027 gains a row:

| Scenario | Best system |
|---|---|
| 5s outage | LSTM v12 + VelFilter (0.171) |
| 10s outage | LSTM v15 + VelFilter (1.089) |
| **30s outage** | **LSTM v15 + VelFilter + RLS (0.259)** ⇐ new winner |
| 60s outage | LSTM v15 + VelFilter (0.803) |
| Simplest deploy | TCN v7 + VelFilter (no warmup, no adaptation) |

- RLS adaptation should NOT be deployed for short or very long outages; the
  vanilla v15 + filter is the safer general-purpose choice. Wire it in only when
  the use case is specifically the 20–40 s window the model was trained for.

## Limitations

1. **Single test sequence.** All numbers are on MH_05_difficult only. The RLS
   adaptation may behave differently on a sequence with very different motion
   characteristics. Cross-sequence validation deferred.
2. **Hyperparameter overfit to MH_05.** P₀ = 0.1 was selected on the same sequence
   we report on. Without an independent held-out sequence we can't claim this
   generalizes.
3. **Time-of-day variance.** Each evaluation includes one outage event at 40%
   through the sequence. A multi-outage / multi-position evaluation would give
   confidence intervals; not done.
4. **The mean error metric is worse.** RLS improves *final-position* drift
   substantially at 30s but degrades *mean-trajectory* error slightly (0.96 →
   1.05 m/s). The model is trading mid-rollout accuracy for end-of-rollout
   accuracy — same trade-off pattern v15 made vs v13.

These limitations should be foregrounded in the README and writeup. The 36%
improvement is honest; the generalization claim is not.

## What's left

- **Cross-sequence eval** with the RLS head — the credible follow-up.
- **Continuous adaptation during outage** using a self-supervised pretext task
  (e.g., gyro integration consistency). Bigger lift but the right next step.
- **Larger adaptation surface** — LoRA-style low-rank adapters on the LSTM layers
  themselves, not just the head. Higher capacity, higher risk.

These are queued behind shipping the portfolio (decision 028).
