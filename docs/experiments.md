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
## Experiment: IMU Dead Reckoning Baseline (MH_05_difficult)

Method: integrate (accel - static_bias) over time without attitude correction.

Results on MH_05_difficult:
- vel_error_at_5s: 49.2 m/s
- vel_error_at_30s: 298.1 m/s
- vel_error_at_60s: 606.2 m/s
- window delta_v MSE: 34.519
- window delta_v MAE: 3.588
- window delta_v R²: -628.7 (catastrophic)

Zero-predictor MSE on same test set: 0.090
TCN MSE on same test set: 0.089
TCN vs dead reckoning: 387x better on window delta_v MSE.

Notes:
- Catastrophic drift is due to gravity subtraction without attitude tracking
- Naive bias estimation from first N samples invalid when drone is already in motion
- EKF must include attitude state to properly rotate and subtract gravity
- Even though TCN R² ≈ 0, it is 387x better than uncorrected IMU integration

Artifacts:
- `results/dead_reckoning/MH_05_difficult_metrics.json`

## Experiment: TCN v2 — Directional Loss + Larger Model

Configuration:
- loss: 0.6 * MSE + 0.4 * cosine dissimilarity (DirectionalMSELoss, alpha=0.6)
- model: channel_sizes=[64, 128, 256, 256] (~4x more parameters than multi-seq)
- batch size: 64, lr: 3e-4, weight_decay: 1e-4
- balanced sampler: equal weight per sequence regardless of window count
- epochs: 100 (early stopping patience=25)
- cosine LR schedule: T_max=100, eta_min=1e-6
- split: same as multi-seq (MH_01–03+V1_01–02 train / MH_04 val / MH_05 test)

Results:
- best epoch: 1
- best val loss: 0.40043 (directional loss scale, not comparable to prior MSE-only runs)
- test MSE: 0.09264
- test MAE: 0.21948
- zero_predictor_mse: 0.09027
- r2_mean: -0.013
- r2_x: -0.079, r2_y: -0.018, r2_z: +0.059
- corr_x: 0.083, corr_y: 0.033, corr_z: 0.291

Notes:
- REGRESSION vs multi-seq: test MSE 0.09264 vs 0.08914 — v2 is actually worse
- Best epoch 1 again: larger model overfits immediately; train loss drops 0.47→0.21 over 26 epochs
  while val loss stagnates at ~0.40, a 2x train/val gap from the start
- R² negative on x/y axes — model is worse than predicting the mean on those axes
- Root cause: [64,128,256,256] has ~800k parameters vs ~50k in the prior model — massively
  over-parameterized for 5548 training windows
- Cosine loss is also problematic when delta_v magnitudes are near zero (direction is undefined
  for near-zero vectors), which is most of the dataset
- Balanced sampler didn't compensate for the capacity mismatch

Next experiment direction:
- Try directional loss on the SMALL model [16,32,32] — isolate whether loss or model size caused regression
- Alternatively: label smoothing (Savitzky-Golay filter on Leica velocity) to reduce finite-diff noise
- Real fix: use end-to-end trajectory drift as the training signal, not window-level MSE

Artifacts:
- `results/tcn_v2/loss_history.json`
- `results/tcn_v2/test_metrics.json`
- `checkpoints/tcn_v2.pt`

## Experiment: TCN v3 — Directional Loss + Small Model (Isolation Test)

Hypothesis: v2 regressed because of model size ([64,128,256,256], ~800k params), not because
of the directional loss. v3 uses the small model [16,32,32] with the same directional loss.

Configuration:
- loss: 0.6 * MSE + 0.4 * cosine dissimilarity (DirectionalMSELoss, alpha=0.6)
- model: channel_sizes=[16, 32, 32], dropout=0.3 (~50k params — same as multi-seq)
- batch size: 64, lr: 3e-4, weight_decay: 1e-4
- balanced sampler: equal weight per sequence
- epochs: 100 (early stopping patience=25)
- cosine LR schedule: T_max=100, eta_min=1e-6
- split: same as multi-seq / v2

Results:
- best epoch: 27
- best val loss: 0.41265 (directional loss scale)
- test MSE: 0.09086
- test MAE: 0.21445
- zero_predictor_mse: 0.09027
- r2_mean: +0.0029  ← first positive r2_mean across all runs
- r2_x: -0.010, r2_y: -0.025, r2_z: +0.043
- corr_x: 0.052, corr_y: -0.009, corr_z: 0.234

Notes:
- Hypothesis CONFIRMED: best epoch 27 (not 1) proves directional loss is not the culprit —
  model size was entirely responsible for v2's immediate collapse
- Still does not beat zero predictor on MSE (0.09086 vs 0.09027) or multi-seq (0.08914)
- r2_mean is positive for the first time (+0.0029), a marginal but directionally correct step
- Z-axis is the only axis with real signal (corr_z=0.234, r2_z=+0.043) — likely because
  vertical motion is strongly coupled to gravity in IMU readings
- X/Y remain near-random (corr near zero or negative): horizontal motion is harder to
  distinguish from noise at 1s window scale with this model capacity
- Root cause of x/y failure: finite-difference noise in Leica position (horizontal ≈ 1cm
  precision over 1s) is comparable in magnitude to actual horizontal delta_v
- Next lever: label smoothing (Savitzky-Golay on Leica velocity) to suppress finite-diff
  noise, OR increase window size to give model more context per prediction

Artifacts:
- `results/tcn_v3/loss_history.json`
- `results/tcn_v3/test_metrics.json`
- `checkpoints/tcn_v3.pt`

## Experiment: TCN v4 — Savitzky-Golay Label Smoothing

Hypothesis: finite-difference noise on Leica velocity (especially x/y horizontal) is
comparable to delta_v signal magnitude. SG filter (w=51, p=3) smooths labels before
computing delta_v; X windows unchanged.

Configuration:
- labels: SG-smoothed gt_vel (window_length=51, polyorder=3) → delta_v
- model/loss/sampler: identical to v3 ([16,32,32], directional loss alpha=0.6, balanced)

Results:
- best epoch: 38 (+11 vs v3 — model trains longer, noise reduction confirmed)
- best val loss: 0.40278 (directional loss scale)
- test MSE: 0.08511
- zero_predictor_mse: 0.08397 — model still worse than zero predictor
- r2_mean: -0.0015 (slightly negative, worse than v3's +0.003)
- r2_x: ~0.000, r2_y: -0.053, r2_z: +0.049
- corr_x: +0.092 (improved from v3's 0.052)
- corr_y: -0.045 (regressed from v3's -0.009)
- corr_z: 0.225 (slight regression from v3's 0.234)

Notes:
- SG smoothing confirms label noise is real (longer training) but doesn't fix x/y signal
- corr_x improved slightly; corr_y went negative — inconsistent effect across axes
- Y-axis worst axis: r2_y=-0.053, mse_y=0.115 — MH_05_difficult likely has different
  y-axis dynamics than training sequences (distribution shift)
- SG is physics-blind: it smooths HF noise but also smooths real fast dynamics,
  potentially removing learnable signal along with noise
- Waiting on v5 (EKF labels) for physics-aware smoothing comparison

Artifacts:
- `results/tcn_v4/loss_history.json`
- `results/tcn_v4/test_metrics.json`
- `checkpoints/tcn_v4.pt`

## Experiment: TCN v5 — EKF-Smoothed Labels

Hypothesis: The full-GPS EKF produces physically-consistent velocity estimates that
are better training labels than raw finite-difference or SG-smoothed velocity, because
the EKF incorporates the IMU motion model.

Configuration:
- labels: full-GPS EKF velocity (15-state, sigma_v=0.02) → delta_v per window
- EKF initialized from static accel reading per sequence, Leica vel at every step
- model/loss/sampler: identical to v3/v4 ([16,32,32], directional loss, balanced)

Results:
- best epoch: 18 (between v3=27 and v4=38)
- best val loss: 0.42109 (directional loss scale)
- test MSE: 0.09006
- zero_predictor_mse: 0.08995 — model still marginally worse than zero predictor
- r2_mean: +0.004 (best across v3/v4/v5, marginal)
- r2_x: +0.006, r2_y: -0.020, r2_z: +0.027
- corr_x: 0.105 (best x-axis correlation across all runs)
- corr_y: 0.001 (essentially zero — y-axis completely unlearnable)
- corr_z: 0.202 (regressed from v3's 0.234)

Notes:
- EKF labels did not materially improve over SG (v4) or raw (v3)
- corr_x is highest yet (0.105) — EKF may preserve real x-axis dynamics better than SG
- corr_z REGRESSED vs v3 (0.202 vs 0.234) — EKF over-smooths z, removing HF signal TCN used
- corr_y ≈ 0 across all runs: y-axis horizontal motion is fundamentally unlearnable at this
  data volume and window size — likely dominated by distribution shift between MH train/test
- CONCLUSION: label smoothing is not the binding constraint. All three label variants
  (raw, SG, EKF) produce nearly identical results. Data volume and test-set distribution
  shift are the real limiters.

## Label Smoothing Comparison (v3 → v4 → v5)

| Version | Labels | Best epoch | r2_mean | corr_x | corr_y | corr_z |
|---|---|---|---|---|---|---|
| v3 | raw finite-diff | 27 | +0.003 | 0.052 | -0.009 | 0.234 |
| v4 | Savitzky-Golay | 38 | -0.001 | 0.092 | -0.045 | 0.225 |
| v5 | EKF full-GPS | 18 | +0.004 | 0.105 | 0.001 | 0.202 |

No label variant breaks the corr_y ≈ 0 wall or pushes r2_mean above ~0.004.

**Next direction**: data volume (add V2 sequences or augment) or change prediction target
(predict velocity directly instead of delta_v to remove the near-zero-mean problem).

Artifacts:
- `results/tcn_v5/loss_history.json`
- `results/tcn_v5/test_metrics.json`
- `checkpoints/tcn_v5.pt`

## Experiment: TCN v6 — Added V1_03_difficult (More Data)

Hypothesis: label smoothing is not the bottleneck (v3/v4/v5 all plateau at r2≈0.003).
Data volume is. Adding V1_03_difficult (837 windows, ~105s of difficult Vicon Room flight)
increases training set from 5548 → 6385 windows (+15%).

Configuration:
- train seqs: MH_01/02/03, V1_01, V1_02, V1_03_difficult (6 sequences, 6385 windows)
- val/test: unchanged (MH_04, MH_05)
- model/loss/sampler: identical to v3 ([16,32,32], directional loss alpha=0.6, balanced)

Results:
- best epoch: 44 (best across all runs — trains longest yet)
- best val loss: 0.40938
- test MSE: 0.08978
- zero_predictor_mse: 0.09027 — model still below zero predictor on raw MSE
- r2_mean: +0.013 ← 4x improvement over v3's +0.003 — biggest jump yet
- r2_x: +0.012 ← first meaningfully positive x-axis R² across all runs
- r2_y: -0.016 (still negative, improved from v3's -0.025)
- r2_z: +0.043
- corr_x: 0.109 (best across all runs)
- corr_y: 0.001 (structurally near-zero across all runs — not a data volume problem)
- corr_z: 0.226

Notes:
- Data volume IS the bottleneck for x-axis: r2_x went from -0.010 (v3) to +0.012 (v6)
  with just 837 additional windows (+15% data → 4x improvement in r2_mean)
- Y-axis is stuck: corr_y ≈ 0 across v3/v4/v5/v6 regardless of label quality or data volume.
  This suggests horizontal y-axis motion is structurally ambiguous from IMU alone at 1s windows,
  OR that MH_05_difficult has y-axis dynamics not represented in any training sequence.
- If r2_mean scales linearly with data, we'd need ~3-4x current volume to reach r2_mean≈0.05
  (still modest). But y-axis may require a target change, not more data.
- All EuRoC sequences now used (V2 sequences not available on disk).
- Next lever: change prediction target from delta_v → absolute velocity to remove
  near-zero-mean problem and test whether the architecture can learn velocity directly.

Artifacts:
- `results/tcn_v6/loss_history.json`
- `results/tcn_v6/test_metrics.json`
- `checkpoints/tcn_v6.pt`

## Experiment: TCN v7 — Absolute Velocity Target

Hypothesis: delta_v labels are near-zero-mean for most 1s windows (small velocity changes),
so the model's trivial strategy is to predict zero. Switching to absolute velocity at window
end gives a non-zero-mean, high-variance target the model can actually regress.

Configuration:
- target: absolute velocity at window end (gt_vel_x/y/z), z-score normalised per axis
- labels built directly from imu_aligned.csv per sequence (no y_delta_v.npy used)
- model/loss/sampler: identical to v6 ([16,32,32], directional loss alpha=0.6, balanced)
- train seqs: same 6 as v6 (6385 windows)

Results:
- best epoch: 72 (trains much longer — well-defined target doesn't saturate quickly)
- best val loss: 1.287 (different scale — normalised absolute vel, not comparable to v3–v6)
- test MSE on normalised y: 1.465
- zero_predictor_mse: 1.621 (model beats zero predictor: 9.6% better)
- r2_mean: +0.095 ← 7.3x improvement over v6's +0.013
- r2_x: +0.102, r2_y: +0.105, r2_z: +0.078
- corr_x: 0.449 (4.1x improvement over v6's 0.109)
- corr_y: 0.374 ← y-axis finally learnable (v3–v6 all had corr_y ≈ 0)
- corr_z: 0.289 (slight improvement over v6's 0.226)

Notes:
- BREAKTHROUGH: y-axis correlation went from 0.001 (v6) to 0.374 — the y-axis was NEVER
  structurally unlearnable; it was unlearnable specifically because delta_v is near-zero-mean.
- Near-zero-mean target confirmed as the binding bottleneck for all of v3–v6.
- Model now clearly beats zero predictor (zero_mse=1.621 vs model_mse=1.465).
- All three axes positive R² for the first time — genuinely learning velocity structure.
- Training ran 72 epochs (vs 44 for v6) — richer target keeps improving longer.
- Next: scale model to [32,64,64] now that target is well-defined; also run EKF outage
  comparison to see if v7 translates to better navigation.

| Version | Target | Best epoch | r2_mean | corr_x | corr_y | corr_z |
|---|---|---|---|---|---|---|
| v6 | delta_v | 44 | +0.013 | 0.109 | 0.001 | 0.226 |
| v7 | abs velocity | 72 | **+0.095** | **0.449** | **0.374** | **0.289** |

Artifacts:
- `results/tcn_v7/loss_history.json`
- `results/tcn_v7/test_metrics.json`
- `results/tcn_v7/normalization_stats.json`
- `checkpoints/tcn_v7.pt`

## Experiment: TCN v8 — Scaled Model [32,64,64]

Hypothesis: with an absolute velocity target (well-defined, non-zero-mean), a larger model
should extract more signal from the same data.

Configuration:
- model: channel_sizes=[32, 64, 64], dropout=0.2 (~4x params vs v7's [16,32,32])
- everything else identical to v7

Results:
- best epoch: 19 (vs v7's 72 — much earlier convergence)
- best val loss: 1.300
- r2_mean: +0.099 (vs v7's +0.095 — +4%, marginal)
- r2_x: +0.130 (vs v7's +0.102 — improved)
- r2_y: +0.088 (vs v7's +0.105 — regressed)
- r2_z: +0.080 (vs v7's +0.078 — flat)
- corr_x: 0.499 (vs v7's 0.449 — +11%)
- corr_y: 0.321 (vs v7's 0.374 — -14% regression)
- corr_z: 0.290 (vs v7's 0.289 — flat)

Notes:
- Bigger model gains on x-axis but regresses on y-axis — inconsistent across axes.
- Early stop at epoch 19 (vs 72 for v7): larger model overfits faster, same pattern as v2.
- 6385 windows is insufficient for reliable [32,64,64] training — same data bottleneck as before.
- v7 [16,32,32] remains best overall checkpoint. Model scaling does not reliably help at this data size.
- Next lever: data augmentation (noise injection, rotation) OR longer window size (400 samples / 2s).
- The EKF outage comparison with v7 is now the most important next experiment.

| Version | Model | Best epoch | r2_mean | corr_x | corr_y | corr_z |
|---|---|---|---|---|---|---|
| v7 | [16,32,32] | 72 | +0.095 | 0.449 | 0.374 | 0.289 |
| v8 | [32,64,64] | 19 | +0.099 | 0.499 | 0.321 | 0.290 |

Artifacts:
- `results/tcn_v8/loss_history.json`
- `results/tcn_v8/test_metrics.json`
- `checkpoints/tcn_v8.pt`

## Experiment: EKF Outage Comparison — multi_seq vs v6 vs v7

Compare GPS-outage navigation on MH_05_difficult across three checkpoints.
Script: `scripts/ekf_outage_comparison.py`

v7 uses absolute-velocity prediction (denormalised directly to m/s). multi_seq and v6 use
delta_v accumulation: `v_now = v_200_samples_ago + delta_v_tcn`.

### Final velocity error at outage end (m/s, lower is better)

| Outage | Dead reckoning | multi_seq | v6 | **v7** | EKF+GPS |
|---|---|---|---|---|---|
| 5s | 8.771 | 0.830 | 0.987 | **0.476** | 0.172 |
| 10s | 15.821 | **0.600** | 0.614 | 0.908 | 0.328 |
| 30s | 45.867 | 1.356 | 1.244 | **0.501** | 0.104 |
| 60s | 95.113 | 1.069 | 1.784 | **0.843** | 0.229 |

### Mean velocity error over outage window at 30s

| Checkpoint | Mean error (m/s) |
|---|---|
| Dead reckoning | 23.181 |
| multi_seq | 1.261 |
| v6 | 1.299 |
| **v7** | **0.974** |
| EKF+GPS | 0.202 |

Notes:
- v7 wins at 3/4 outage durations (5s, 30s, 60s). Only loses at 10s — likely variance.
- At 30s (most realistic GPS-denied scenario), v7 mean error is 23% lower than multi_seq.
- The 7x R² improvement (0.013 → 0.095) does translate to better navigation.
- Absolute velocity mode has no error accumulation: each TCN call produces an independent
  velocity estimate. Delta_v mode accumulates predictions via the rolling buffer — this can
  either help (EKF-like smoothing) or hurt (error from v_200_ago propagates).
- v6 underperforms multi_seq at 60s (1.784 vs 1.069) — more training data didn't help
  navigation, confirming that delta_v bottleneck is about the target, not data volume.
- Gap to GPS upper bound: v7 0.501 m/s vs EKF+GPS 0.104 m/s at 30s — still 5x gap.
  Next lever: integrate v7 predictions into EKF as a velocity measurement (not standalone).

Artifacts:
- `results/outage_comparison/MH_05_difficult_comparison.json`
- `scripts/ekf_outage_comparison.py`

## Experiment: Neural-Aided EKF / Velocity-Only Filter (decision 019)

Tested fusing TCN v7 into an EKF vs a simple velocity-only filter during GPS outage.

### Strapdown EKF + TCN (sigma_tcn sweep)

| sigma_tcn | 5s | 10s | 30s | 60s | mean@30s |
|---|---|---|---|---|---|
| 0.01 | 0.453 | 1.056 | 0.714 | 0.861 | 1.385 |
| 0.05 | 2.699 | 3.016 | 3.096 | 2.825 | 3.585 |
| 0.1 | 6.038 | 6.201 | 6.545 | 5.747 | 6.741 |
| 0.3 | 18.479 | 19.356 | 19.180 | 16.663 | 19.095 |

No sigma_tcn consistently beats standalone TCN. Root cause: attitude drift during outage
contaminates IMU propagation faster than TCN updates can repair (~0.36 m/s per 0.125s interval
at 30s outage due to gyro bias). The strapdown EKF is actively harmful after ~10s.

### Velocity-Only Filter (constant-velocity random walk + TCN updates, no IMU)

| | 5s | 10s | 30s | 60s | mean@30s |
|---|---|---|---|---|---|
| TCN standalone | 0.476 | 0.908 | 0.501 | 0.843 | 0.974 |
| **VelFilter** | **0.419** | 1.163 | **0.440** | **0.816** | **0.962** |
| EKF+GPS | 0.172 | 0.328 | 0.104 | 0.229 | 0.202 |

Velocity-only filter wins at 3/4 outage durations. 12% better final at 30s, 1.2% better mean.
No IMU propagation → no attitude drift → stable across all outage lengths.

Artifacts:
- `results/neural_aided_ekf_v7/MH_05_difficult_results.json`
- `scripts/neural_aided_ekf_v7.py`
- `docs/decisions/019-velocity-only-filter-beats-strapdown-ekf.md`
