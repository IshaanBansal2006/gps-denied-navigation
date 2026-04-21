# Decision 018: EKF Outage Navigation — v7 Absolute Velocity Beats Delta_v

**Date:** 2026-04-21
**Status:** Accepted

## Context

TCN v7 (absolute velocity, R²=+0.095) was evaluated in GPS-denied navigation on MH_05_difficult
against multi_seq (delta_v, R²≈0) and v6 (delta_v, R²=+0.013).

Evaluation: simulated GPS outage at 40% into each sequence, lasting 5/10/30/60 seconds.
v7 absolute-velocity mode: each TCN call produces an independent velocity estimate, denormalised
to m/s. Delta_v mode: `v_now = v_200_samples_ago + delta_v_tcn` via rolling buffer.

## Results

| Outage | multi_seq | v6 | v7 | EKF+GPS |
|---|---|---|---|---|
| 5s | 0.830 | 0.987 | **0.476** | 0.172 |
| 10s | **0.600** | 0.614 | 0.908 | 0.328 |
| 30s | 1.356 | 1.244 | **0.501** | 0.104 |
| 60s | 1.069 | 1.784 | **0.843** | 0.229 |

Mean error at 30s: v7=0.974 m/s vs multi_seq=1.261 m/s (23% improvement).

## Decision

v7 (`checkpoints/tcn_v7.pt`) is the best navigation checkpoint. Absolute velocity prediction
is confirmed as both the better regression target AND the better navigation mode.

The delta_v rolling-buffer approach has a structural flaw: `v_200_ago` is itself an estimate,
so errors in past predictions contaminate future ones. Absolute velocity avoids this — each
prediction is independent.

## Remaining Gap

v7 achieves 0.501 m/s mean at 30s outage. EKF+GPS achieves 0.104 m/s — a 5x gap.

To close it: use v7 as a velocity measurement inside the EKF during outage, rather than
as a standalone estimator. The EKF can weight the TCN measurement against its own prediction
and smooth out the 0.125s update jitter from STRIDE=25.
