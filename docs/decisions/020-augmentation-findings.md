# Decision 020: Data Augmentation — What Works and What Doesn't

**Date:** 2026-04-21
**Status:** Accepted

## Context

After v7 established R²=+0.095 as the baseline, the goal was to improve via augmentation
to expand effective training set size beyond the 6385 available windows.

Two approaches tested:

## TCN v9: Yaw Rotation + Noise + Reflection

Applied to normalized data: random yaw rotation R ∈ SO(2) × {0} to all 3 gyro channels,
all 3 accel channels, and velocity label simultaneously. Also added Gaussian noise and
random x/y reflection.

**Results**: R² collapsed from +0.095 → +0.005 (19x worse). corr_y went negative (-0.103).

**Root cause**: EuRoC sequences have heading-specific dynamics. Machine Hall flies along
corridors with a consistent heading; Vicon Room has another fixed heading. The model in v7
was exploiting heading-specific correlations ("when IMU looks like corridor flight, velocity
is roughly northward") — the exact kind of prior that correctly generalizes to MH_05_difficult.
Yaw rotation augmentation destroyed this prior: the model now sees all headings uniformly and
cannot learn heading-specific features. At test time (fixed heading distribution), this is
purely harmful. Augmentation mismatched the train-augmented and test distributions.

Secondary issue: applying rotation to per-channel-normalized data is incorrect when channels
have different per-axis means (the EuRoC-specific heading is encoded in those means). Rotation
after normalization introduces physically inconsistent samples.

## TCN v10: Noise Only (σ=0.05 in normalized space)

Added Gaussian noise with σ=0.05 to all 6 normalized IMU channels. No rotation, no reflection.

**Results**: R² +0.0979 vs v7's +0.095 — essentially identical. corr_x +4%, corr_y -5%, corr_z +3%.

**Root cause**: The train/val gap (train≈1.0, val≈1.28) is not driven by overfitting to
specific training samples — it's a systematic distribution shift between train sequences
(MH_01–03, V1_01–03) and val (MH_04_difficult). Noise regularization cannot address this.
The model is not memorizing training windows; it's failing to generalize across sequences
with different dynamics.

## Decision

Standard data augmentation does not improve EuRoC velocity prediction because:
1. The test distribution is heading-specific (fixed EuRoC corridors) — heading augmentation
   destroys useful priors rather than generalizing them.
2. The train/val gap is due to sequence-level distribution shift, not sample-level overfitting.
   Noise regularization cannot fix sequence-level shift.

**Correct next levers**:
1. Longer window (400 samples / 2s): more temporal context → more motion patterns visible.
2. Different architecture (LSTM/Transformer): maintains state across windows.
3. End-to-end navigation loss: train directly on EKF drift rather than per-window velocity MSE.
