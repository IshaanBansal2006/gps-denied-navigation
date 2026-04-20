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

## Experiment: TCN Multi-Sequence (7 sequences, corrected splits)

Configuration:
- window size: 200
- stride: 25
- features: gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
- target: delta velocity
- epochs: 100 (early stopping patience=20)
- batch size: 32
- learning rate: 3e-4 with ReduceLROnPlateau (factor=0.5, patience=8)
- optimizer: Adam, weight_decay=1e-4
- loss: MSE
- model: channel_sizes=[16,32,32], dropout=0.3

Split (cross-sequence, both environments in train):
- train: MH_01_easy (1465), MH_02_easy (1204), MH_03_medium (1052), V1_01_easy (1150), V1_02_medium (677) → 5548 windows
- val: MH_04_difficult (790 windows)
- test: MH_05_difficult (888 windows)

Sequences processed: all 7 available EuRoC bags extracted via ROS Noetic rosbag.
MH_01–05 from machine_hall, V1_01–02 from vicon_room1.

Results:
- best epoch: 11
- best val loss: 0.10577
- test MSE: 0.08914
- test MAE: 0.21136

Axis-wise:
- mse_x: 0.09958
- mse_y: 0.11557
- mse_z: 0.05227
- mae_x: 0.23029
- mae_y: 0.23584
- mae_z: 0.16796

Notes:
- NOT directly comparable to prior runs — val/test use MH_04/05 (difficult), prior runs used a
  chronological split of MH_01 only. Difficult sequences have genuinely different dynamics.
- Zero-prediction baseline MSE on this test split: 0.09027 (our model: 0.08914 — only 1.2% better)
- Best epoch improved from 4 (single-seq) to 11 (multi-seq), confirming reduced overfitting
- Absolute MSE improvement over the same test set zero-predictor is marginal — model is not yet
  learning meaningful velocity change structure across sequences
- Root cause: delta_v labels are near-zero mean, finite-difference noise dominates signal
- Prior "improved" run test MSE (0.04794) was actually WORSE than its zero-pred baseline (0.03842),
  so the multi-sequence run is the first one that at least doesn't regress below zero-prediction
- Next lever: label smoothing or use GT velocity directly as auxiliary supervision; alternatively
  switch to predicting cumulative velocity rather than delta_v

Artifacts:
- `results/tcn_multi/loss_history.json`
- `results/tcn_multi/test_metrics.json`
- `checkpoints/tcn_multi.pt`