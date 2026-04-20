# Experiments

## Goals

Evaluate:

- Motion prediction accuracy
- Drift reduction
- GPS denied performance

## Metrics

Position error

Velocity error

Drift rate

## Experiments

### Baseline

IMU only

### Neural Model

IMU + Neural Prediction

### EKF

IMU + EKF

### Neural-Aided EKF

IMU + Neural + EKF

## Evaluation

Compare drift during GPS outage.

## Experiment: First full TCN baseline

Configuration:
- window size: 200
- features: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
- target: delta velocity
- epochs: 50
- batch size: 64
- optimizer: Adam
- loss: MSE

Results:
- best epoch: 1
- best val loss: 0.04398194
- test MSE: 0.05011975
- test MAE: 0.16644402

Axis-wise:
- mse_x: 0.02855728
- mse_y: 0.05025956
- mse_z: 0.07154243
- mae_x: 0.11777277
- mae_y: 0.17699020
- mae_z: 0.20456913

Notes:
- full pipeline worked end-to-end
- overfitting began almost immediately
- this run is the baseline to beat
- Loss curve inspection showed that validation loss was best at epoch 1 and did not improve afterward, confirming immediate overfitting in the first full baseline run.

Artifacts:
- `results/tcn_baseline/loss_history.json`
- `results/tcn_baseline/test_metrics.json`
- `results/tcn_baseline/loss_curve.png`

## Experiment: TCN Improved (anti-overfitting)

Configuration:
- window size: 200
- stride: 25 (down from 1)
- features: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
- target: delta velocity
- epochs: 50 (early stopping patience=10)
- batch size: 64
- optimizer: Adam, weight_decay=1e-4
- loss: MSE
- model: channel_sizes=[16,32,32], dropout=0.3

Results:
- best epoch: 4
- best val loss: 0.03841785
- test MSE: 0.04794239
- test MAE: 0.15808906

Axis-wise:
- mse_x: 0.01750548
- mse_y: 0.04470174
- mse_z: 0.08161996
- mae_x: 0.10719377
- mae_y: 0.16244704
- mae_z: 0.20462640

Notes:
- Training stopped early at epoch 14 (best at epoch 4)
- Reducing stride from 1 → 25 cut window count from ~180k to 1465
- Smaller model capacity + higher dropout pushed best epoch from 1 to 4
- Val loss improved ~12.7% over baseline (0.03842 vs 0.04398)
- Test MSE improved ~4.3% over baseline (0.04794 vs 0.05012)
- Fundamental bottleneck is single short flight sequence; multi-sequence training is the next lever

Artifacts:
- `results/tcn_improved/loss_history.json`
- `results/tcn_improved/test_metrics.json`
- `checkpoints/tcn_improved.pt`

## Experiment: TCN Multi-Sequence (6 sequences)

Configuration:
- window size: 200
- stride: 25
- features: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
- target: delta velocity
- epochs: 50 (early stopping patience=10)
- batch size: 64
- optimizer: Adam, weight_decay=1e-4
- loss: MSE
- model: channel_sizes=[32,64,64], dropout=0.2

Split:
- train: MH_01_easy (1465), MH_02_easy (1204), MH_03_medium (1052), V1_01_easy (1150), V1_02_medium (677) → 5548 windows
- val: MH_04_difficult (790 windows)
- test: MH_05_difficult (888 windows)

Results:
- best epoch: 3
- best val loss: 0.10596438
- test MSE: 0.36955845
- test MAE: 0.43398896

Axis-wise:
- mse_x: 0.38278913
- mse_y: 0.60909671
- mse_z: 0.11678942
- mae_x: 0.47371766
- mae_y: 0.57124698
- mae_z: 0.25700217

Notes:
- NOTE: NOT directly comparable to prior runs — val/test sequences are "difficult" (MH_04, MH_05),
  which have larger delta velocities and harder flight dynamics by design
- Split was adjusted from original plan (V1_02 moved to train; MH_05_difficult used as test)
  to avoid cross-environment distribution shift masking model quality
- Train windows increased ~3.8x (1465 → 5548) but best epoch regressed to 3 (from 4)
- Large val→test gap (0.106 → 0.370) indicates model does not generalize from MH_04 difficulty
  to MH_05 difficulty — different maneuver patterns within "difficult" tier
- Y-axis error is worst (mse_y=0.609), suggesting poor lateral motion generalization on aggressive flights
- Absolute MSE numbers inflated by difficult sequence scale — compare to dead reckoning baseline next

Next step: implement IMU dead reckoning baseline to establish whether TCN MSE (even at 0.37)
beats uncorrected IMU integration on difficult sequences.

Artifacts:
- `results/tcn_multi/loss_history.json`
- `results/tcn_multi/test_metrics.json`
- `checkpoints/tcn_multi.pt`