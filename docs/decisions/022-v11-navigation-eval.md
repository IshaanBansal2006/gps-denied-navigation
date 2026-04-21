# Decision 022: TCN v11 Navigation Eval — Better R² Doesn't Fix 30s Navigation

**Date:** 2026-04-21
**Status:** Accepted

## Context

TCN v11 improved R² by 66% over v7 (0.095 → 0.158) via 2x longer window and 6-layer
TCN (RF=29 → 253 samples). The navigation eval re-runs the velocity-only filter and
strapdown EKF comparison using the v11 checkpoint.

## Results (MH_05_difficult)

### Final velocity error (m/s, lower is better)

| Outage | Dead reckoning | **v7 standalone** | **v11 standalone** | **v7 VelFilter** | **v11 VelFilter** | EKF+GPS |
|---|---|---|---|---|---|---|
| 5s  | 8.771  | 0.476 | **0.330** | 0.419 | **0.284** | 0.172 |
| 10s | 15.821 | 0.908 | 1.346     | 1.163 | 1.315     | 0.328 |
| 30s | 45.867 | 0.501 | 0.546     | **0.440** | 0.485 | 0.104 |
| 60s | 95.113 | 0.843 | **0.740** | 0.816 | **0.769** | 0.229 |

### Mean velocity error at 30s outage

| Mode | v7 | v11 | Delta |
|---|---|---|---|
| TCN standalone | 0.974 | 0.973 | −0.1% |
| VelFilter | 0.962 | **0.955** | −0.7% |
| EKF+GPS | 0.202 | — | — |

## Interpretation

**Short outages (5s): v11 wins convincingly.** Standalone −31%, VelFilter −32%. The 66% R²
improvement directly translates at short outages where individual prediction accuracy
matters most and the filter hasn't had time to accumulate state.

**Medium-long outages (10s, 30s): v11 is slightly worse.** At 30s VelFilter: 0.485 vs 0.440
(+10%). Two mechanisms:

1. **Cold-start delay**: v11's 400-sample window means no TCN output for the first 2s
   of an outage (vs 1s for v7). During this gap the filter drifts on its process noise
   alone. For a 30s outage this 1s gap is small, but the momentum can matter.

2. **Filter saturation**: at 30s, the VelFilter already smooths per-window noise
   effectively. Marginal R² improvements (0.095 → 0.158) don't reduce the dominant
   error source, which is sequence-level distribution shift between train and test.
   The filter is already extracting most of the useful signal from v7's predictions;
   reducing per-window variance by 66% gives diminishing returns.

**Long outages (60s): v11 wins.** Standalone −12%, VelFilter −6%. Sustained long-horizon
accuracy matters more at 60s, where v11's larger RF pays off.

## Decision

**v7 remains the best checkpoint for the 30s outage scenario** (the target use case).
v11 is better for short (5s) and long (60s) outages. The velocity-only filter at 30s
is currently bottlenecked by sequence-level distribution shift, not per-window TCN accuracy.

The remaining gap to GPS (0.440 → 0.104 m/s at 30s) cannot be closed by further TCN
accuracy improvements alone. The filter architecture itself needs to change.

## Revised Next Levers

1. **LSTM/Transformer**: state persists across windows — avoids cold-start, can smooth
   over prediction errors, and handles the 10s regime where both standalone and filter
   v11 regress vs v7.
2. **End-to-end navigation loss**: train directly on trajectory drift over the 30s outage
   window rather than per-window velocity MSE. This directly optimizes the metric that
   matters and will expose the distribution shift as a training signal.
3. **More diverse training data**: sequence-level shift (MH_01-03+V1_01-03 vs MH_04/05)
   is the dominant error. V2 sequences or synthetic augmentation of trajectory-level
   dynamics could help.
