# Decision 023: LSTM v12 — Persistent State Improves X/Y but Z Regresses

**Date:** 2026-04-21
**Status:** Accepted

## Context

TCN v11 showed that temporal context matters (R² +66% over v7), but the velocity-only
filter at 30s was slightly worse — the bottleneck was statelessness, not just receptive
field. An LSTM addresses this directly: hidden state (h, c) carries forward indefinitely,
eliminating the cold-start and providing true sequence memory.

## Architecture

- 2-layer LSTM, hidden_size=128, dropout=0.3 (~264k params vs v11's ~31k)
- Dense supervision: predict absolute velocity at ALL 400 timesteps per chunk (not just end)
  → 400x more training signal per chunk than TCN
- Chunk length=400 (2s), stride=200 (50% overlap)
- Same DirectionalMSELoss(alpha=0.6), Adam lr=3e-4, cosine schedule

## Results

| Metric | v7 (TCN, 1s) | v11 (TCN, 2s) | **LSTM v12** | Delta vs v11 |
|---|---|---|---|---|
| r2_mean | +0.095 | +0.158 | **+0.203** | **+28%** |
| r2_x | +0.102 | +0.150 | **+0.288** | **+92%** |
| r2_y | +0.105 | +0.221 | **+0.270** | +22% |
| r2_z | +0.078 | +0.102 | +0.050 | **−51%** |
| corr_x | 0.449 | 0.427 | **0.543** | +27% |
| corr_y | 0.374 | 0.486 | **0.556** | +14% |
| corr_z | 0.289 | 0.334 | 0.253 | **−24%** |
| best_val | 1.287 | 1.230 | **1.200** | −2.4% |
| best_epoch | 72 | 35 | 53 | — |
| train/val gap | — | 1.68x | 1.69x | flat |

## Interpretation

**X/Y horizontal velocity: clear win.** corr_x +27%, corr_y +14% over v11. r2_x doubled
vs v11. Horizontal motion has sustained, structured dynamics over 2s+ that the LSTM
exploits via its recurrent memory. Dense supervision means the LSTM receives gradient
signal from every step, forcing it to track velocity continuously rather than just at
chunk boundaries.

**Z vertical velocity: significant regression.** r2_z dropped from +0.102 (v11) to +0.050
(−51%). corr_z 0.334 → 0.253 (−24%).

Root cause: vertical motion in EuRoC is impulsive — altitude changes happen briefly and
return to near-zero quickly. The dense loss assigns equal weight to all 400 timesteps,
most of which are z≈0 (constant altitude). The LSTM learns to predict "z near its mean"
aggressively. At the rare timesteps with actual vertical motion, this is wrong. The TCN
window-end prediction only had to be right at one moment; the LSTM has to be right at
every moment including the boring flat ones, which biases it away from z-dynamics.

Additionally, vertical velocity may have more sequence-level distribution shift between
training sequences (indoor machine hall, different hover heights) and MH_05_difficult.

**Train/val gap**: still 1.69x, same as v11. Sequence-level distribution shift persists —
the LSTM hasn't solved it. Adding more parameters doesn't close this gap.

## Decision

LSTM v12 is the best overall model by r2_mean (+0.203). Use as the new training baseline.

For navigation, the z-axis regression is a concern — the velocity-only filter uses all
3 axes. A dedicated nav eval (decision 024) will determine if the x/y gains outweigh
the z loss at 30s.

## Next Levers

1. **Navigation eval with LSTM v12** — does x/y improvement outweigh z regression at 30s?
2. **Weighted loss**: upweight z-axis prediction, or use per-axis loss coefficients.
3. **End-to-end navigation loss**: bypass per-axis accuracy entirely; train on 30s drift
   directly. Would naturally discover the right axis weightings for navigation.
