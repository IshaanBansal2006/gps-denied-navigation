# Decision 031: Test-Time Training (TTT) Adapter — Wash on EuRoC

**Date:** 2026-05-24
**Status:** Accepted (negative result)

## Context

Decision 029 noted that the RLS adaptation head only updates the final linear
projection, leaving the LSTM body frozen. The natural next attempt is
Test-Time Training (TTT): run K gradient steps over (IMU, GPS-aided-velocity)
pairs from the pre-outage window using the actual training loss — this
adapts the *whole* model, not just the head.

Pre-outage warmup gives ~10–60 s of GPS-aided velocity targets we can use as
inner-loop training data. Sun et al. 2020 (the canonical TTT paper) found
auxiliary self-supervised pretext tasks help in classification settings;
here the supervisor is the GPS-aided velocity itself, so no pretext task is
needed.

## Approach

`gps_denied_nav/adaptation/ttt.py` — `TTTAdapter` class with a context-manager
adapt() that snapshots model weights, runs K gradient steps with Adam, then
restores the snapshot on exit. Trainable parameter mask can freeze the first
N LSTM layers (`freeze_lstm_layers`) to preserve low-level IMU features.

Inner-loop loss: per-sample velocity MSE on 1-second IMU windows sampled
from the pre-outage region. Simpler than the full differentiable-filter nav
loss and ~50× faster — appropriate for inference-time adaptation.

## Sweep on val MH_04_difficult (30 s outage at start_frac=0.4)

| Config | final_vel (m/s) | final_pos (m) |
|---|---:|---:|
| Vanilla v15 + filter (baseline) | **0.7820** | 17.15 |
| v15 + filter + RLS (decision 029 baseline) | **0.7495** | 26.00 |
| TTT K=3,  lr=5e-5 | 0.8519 | 15.46 |
| TTT K=3,  lr=1e-5 | 0.8282 | 16.51 |
| TTT K=5,  lr=1e-5 | 0.8337 | 16.29 |
| TTT K=5,  lr=3e-6 | 0.8224 | 16.69 |
| TTT K=10, lr=5e-5 | 0.8719 | **14.43** |
| TTT K=10, lr=1e-5 | 0.8440 | 15.80 |
| TTT K=15, lr=1e-5 | 0.8514 | 15.48 |
| TTT K=15, lr=3e-6 | 0.8323 | 16.35 |
| TTT K=30, lr=5e-5 | 0.8071 | 17.15 |
| TTT K=30, lr=1e-5 | 0.8627 | 14.90 |
| TTT K=3,  lr=1e-5 + RLS | 0.7624 | 26.06 |
| TTT K=10, lr=5e-5 + RLS | 0.7824 | 25.67 |
| TTT K=10, lr=1e-5 + RLS | 0.7732 | 26.04 |
| TTT K=30, lr=5e-5 + RLS | 0.7898 | 25.68 |
| TTT K=30, lr=1e-5 + RLS | 0.7816 | 25.66 |

## Findings

### TTT alone makes final velocity *worse*
- No TTT configuration beats vanilla v15 on `final_velocity_error`.
- The smallest degradation is at K=30 lr=5e-5 (0.81 vs vanilla 0.78, +3%).
- TTT does *modestly* improve `final_position_drift` (14.43 m vs vanilla
  17.15 m at K=10 lr=5e-5, -16%) — likely because the slightly-biased
  velocity predictions cancel out over the rollout.

### TTT + RLS combo regresses
- TTT alone: best final_vel 0.81.
- v15 + RLS alone: best final_vel 0.75.
- TTT + RLS combo: best final_vel 0.76 — same as RLS alone, but with the
  position-drift cost of RLS (~26 m).
- TTT and RLS adapt the same target (matching GPS-aided velocity on
  pre-outage windows). They redundantly fight for the same fit and don't
  combine additively.

### Why TTT doesn't help here
The v15 LSTM was trained on the EuRoC training set, of which MH_04 is the
val sequence held out from the same distribution. Pre-outage IMU windows on
MH_04 are *in-distribution* to the training set — there's no domain shift
for TTT to correct. K gradient steps on in-distribution data don't move
the model meaningfully (loss is already near the converged value).

For TTT to win, the test-time data must look qualitatively different from
training (e.g., a different sensor, a different vehicle, a different
environment). Cross-dataset eval on TUM-VI or KITTI would be the right test;
deferred per decision 028 sprint scope.

## Decision

TTT does **not** ship to the README headline. The TTT adapter is included
in the package (`gps_denied_nav.adaptation.TTTAdapter`) and documented for
future cross-dataset experiments where domain shift gives the method
something to work on.

## What this changes about the project

- New module: `gps_denied_nav/adaptation/ttt.py` (~120 lines).
- New eval script: `scripts/neural_aided_ekf_lstm_v15_ttt.py`.
- No headline change. v15 + filter + RLS remains the best 30-s system on
  EuRoC MH_05 (decision 029).

## What's left

If RLS adaptation continues to be the only mechanism that helps, the next
attack is **continuous adaptation during the outage** (decision 032
pending) — the regime where the model genuinely sees something it wasn't
trained on (its own predictions accumulating error over 30 seconds).
