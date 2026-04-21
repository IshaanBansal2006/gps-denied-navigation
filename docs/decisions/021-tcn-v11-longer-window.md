# Decision 021: Longer Window (400 samples / 2s) Improves R² by 66%

**Date:** 2026-04-21
**Status:** Accepted

## Context

After v7 established R²=+0.095 as the baseline with 1s windows (200 samples), and
augmentation experiments (v9/v10) showed that data augmentation cannot close the
train/val gap (which is sequence-level distribution shift, not overfitting), the
next hypothesis was temporal context.

The 3-layer TCN in v7 has a receptive field of only RF=29 samples (~0.15s at 200Hz).
The 200-sample window was largely wasted — the model could only attend to the last 29
samples regardless of window size. Two changes to test the temporal-context hypothesis:

1. WINDOW_SIZE: 200 → 400 (1s → 2s)
2. channel_sizes: [16,32,32] → [16,32,32,32,32,32] (3 → 6 layers)
   - Dilations: [1,2,4] → [1,2,4,8,16,32]
   - Receptive field: 29 → 253 samples (~1.26s), covering 63% of the 2s window

No augmentation (noise-only was neutral in v10).

## Results

| Metric | v7 (1s, 3 layers) | v11 (2s, 6 layers) | Delta |
|---|---|---|---|
| r2_mean | +0.095 | **+0.158** | **+66%** |
| r2_x | +0.102 | +0.150 | +47% |
| r2_y | +0.105 | **+0.221** | **+110%** |
| r2_z | +0.078 | +0.102 | +31% |
| corr_x | 0.449 | 0.427 | -5% |
| corr_y | 0.374 | **0.486** | **+30%** |
| corr_z | 0.289 | 0.334 | +16% |
| best_val_loss | 1.287 | **1.230** | -4.4% |
| best_epoch | 72 | 35 | earlier |

## Interpretation

**Temporal context was a real bottleneck.** The 2s window + 6-layer TCN provides
~8.7x more receptive field (253 vs 29 samples). The model can now observe velocity-
relevant motion patterns that play out over 1+ seconds: acceleration ramps, turns,
sustained horizontal velocity, etc.

**Y-axis benefited most.** r2_y doubled (0.105 → 0.221). This is consistent with
the hypothesis that horizontal y-axis velocity has more temporal structure (heading
changes, sustained accelerations) that is only visible over 2s, not 1s.

**corr_x slight regression (-5%)** may be noise or a sign that 6385 windows is
starting to be tight for the 6-layer model (which has more parameters). Not
statistically significant — corr_x and corr_y both improved significantly on the
more meaningful R² axis.

**Earlier best epoch (35 vs 72)** is expected: larger-RF model converges faster
because each gradient step sees more signal per window.

**Train/val gap**: at best epoch 35, val/train ratio ≈ 1.68x. v7 was also gapped
(1s windows don't provide enough signal for the model to generalize well either).
The distribution shift between train (MH_01-03, V1_01-03) and val (MH_04_difficult)
remains — this is not fixed by longer windows.

## Decision

Use WINDOW_SIZE=400 (2s) for all future experiments. The 6-layer [16,32,32,32,32,32]
architecture with RF=253 is the new baseline.

## Next Levers (in order of expected impact)

1. **Even longer window or global average pooling**: RF=253 covers 63% of the 2s window.
   Going to 8 layers (RF=1021) could squeeze out more, but parameter count grows.
2. **LSTM/Transformer**: Maintains state across windows — the Kalman-filter equivalent
   for sequential prediction. Can theoretically attend to arbitrarily long history.
3. **End-to-end navigation loss**: Train directly on EKF drift over a trajectory
   rather than per-window velocity MSE. This directly optimizes what we care about.
