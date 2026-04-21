# Decision 017: Switch Prediction Target to Absolute Velocity (TCN v7)

**Date:** 2026-04-21
**Status:** Accepted

## Context

TCN v3–v6 all predicted delta_v (change in velocity over a 200-sample window). Despite
varying label quality (raw, SG-smoothed, EKF-smoothed) and adding all available EuRoC
sequences, R² stalled at +0.013 and corr_y remained at ≈0.001 across every run.

The hypothesis: delta_v ≈ 0 for most 1s windows during smooth flight. The trivial
zero-predictor has near-zero MSE on near-zero-mean labels, so MSE loss provides almost
no gradient signal. The y-axis is worst because horizontal velocity changes are smallest.

## Decision

Predict absolute velocity at the end of the window (`gt_vel_x/y/z` from imu_aligned.csv)
instead of `delta_v = vel[end] - vel[start]`.

Labels are z-score normalised per axis using train-split statistics before training.
Everything else (model, loss, sampler, splits) identical to v6.

## Results (TCN v7)

| Metric | v6 (delta_v) | v7 (abs vel) | Δ |
|---|---|---|---|
| r2_mean | +0.013 | **+0.095** | +7.3x |
| corr_x | 0.109 | **0.449** | +4.1x |
| corr_y | 0.001 | **0.374** | +374x |
| corr_z | 0.226 | **0.289** | +28% |
| best epoch | 44 | 72 | trains longer |

## Consequences

- y-axis is now learnable: 0.374 vs ≈0 across all v3–v6. This proves y-axis motion is NOT
  structurally ambiguous from IMU — it was just unlearnable from near-zero-mean labels.
- All three axes now show positive R² — first time in this project.
- Model clearly beats zero predictor (MSE 1.465 vs zero_pred 1.621 on normalised y).
- delta_v target is retired. All future TCN experiments should use absolute velocity.
- EKF integration during GPS outage needs updating: instead of adding delta_v to EKF state,
  use TCN-predicted velocity directly as a velocity measurement in the EKF update step.
- Next: scale model to [32,64,64] (v8) since target is now well-defined.
