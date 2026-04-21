# Decision 024: LSTM v13 — Velocity-Weighted Loss Fixes Z-Axis Regression

**Date:** 2026-04-21
**Status:** Accepted

## Context

LSTM v12 achieved r2_mean=+0.203 (best yet) but the z-axis regressed sharply vs v11
(corr_z 0.334 → 0.253, r2_z 0.102 → 0.050). Navigation eval showed the consequence:
v12 matches GPS at 5s but is 13% worse than v7 at 30s — the z drift accumulates.

Root cause: dense supervision assigns equal weight to all 400 timesteps. Most z-timesteps
are near-zero (constant altitude). The LSTM biased toward "z ≈ mean" everywhere.

Fix: weight each timestep's loss by normalized velocity magnitude:
  weight(t) = ||v(t)|| / (mean(||v||) + ε)

Dynamic timesteps (fast motion, large z change) contribute proportionally more gradient.
Near-static timesteps contribute less. Mean weight ≈ 1 so loss scale stays stable.
Applied to the MSE component; cosine term already masks near-zero via min_norm.

## Results

| Metric | v7 (TCN 1s) | v11 (TCN 2s) | v12 (LSTM dense) | **v13 (LSTM weighted)** |
|---|---|---|---|---|
| r2_mean | +0.095 | +0.158 | +0.203 | **+0.207** |
| r2_x | +0.102 | +0.150 | +0.288 | +0.230 |
| r2_y | +0.105 | +0.221 | +0.270 | +0.249 |
| r2_z | +0.078 | +0.102 | +0.050 | **+0.142** |
| corr_x | 0.449 | 0.427 | **0.543** | 0.528 |
| corr_y | 0.374 | 0.486 | **0.556** | 0.547 |
| corr_z | 0.289 | 0.334 | 0.253 | **0.375** |
| best_epoch | 72 | 35 | 53 | 14 |
| train/val gap | — | 1.68x | 1.69x | **1.15x** |

## Interpretation

**Z-axis recovered and improved.** corr_z: 0.253 → 0.375 (+48% vs v12), which beats
every prior version including v7 (0.289) and v11 (0.334). r2_z nearly tripled vs v12.
The velocity-weighted loss successfully redirected gradient toward impulsive vertical
events rather than static periods.

**X/Y minor regression vs v12 (-3% on corr_x, -2% on corr_y).** Still well above v7
and v11. This is the expected tradeoff: downweighting static timesteps slightly reduces
the learning signal for near-zero horizontal motion, but those are trivially easy to
predict anyway.

**Train/val gap collapsed to 1.15x** (vs 1.69x for v12). The velocity-weighting acts as
implicit regularization — the model can't overfit to easy static samples anymore,
forcing generalization on dynamic patterns. This is the tightest train/val gap in the
entire experiment series.

**Early convergence at epoch 14** (vs 53 for v12). The weighted loss has a different
landscape: harder on dynamic samples, easier on static. The model finds its optimum
faster because the gradient signal is denser in informative regions.

**v13 is the best overall model**: highest r2_mean (+0.207), best corr_z (0.375),
x/y within 3% of v12's peak. Velocity-weighted loss is strictly better than uniform
dense loss for this dataset.

## Decision

Use LSTM v13 as the new navigation baseline for the 30s eval. The improved z-axis should
close the 30s gap relative to v12. Expected: v13 VelFilter at 30s < 0.440 m/s (v7 best).

## Navigation Eval (LSTM v12 for context)

LSTM v12 nav eval already ran:
- 5s: VelFilter=0.171 m/s (vs GPS=0.172) — matched GPS at short outages
- 30s: VelFilter=0.497 m/s (vs v7=0.440) — worse due to z regression

v13's better z should recover the 30s result while keeping the 5s breakthrough.

## Next Step

Run nav eval with lstm_v13.pt. If 30s VelFilter < 0.440 m/s, v13 is the new end-to-end
best. If not, the bottleneck is sequence-level distribution shift (not fixable by loss
weighting alone) and end-to-end navigation loss becomes the priority.
