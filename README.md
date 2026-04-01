## Neural-Aided GPS-Denied Navigation for UAVs

### Project goal

Build a neural-aided inertial navigation pipeline for GPS-denied UAV navigation.

Current scope: use fixed-length windows of IMU data as input and predict **delta velocity** \(\Delta v\) across the window. The intended next stage (not implemented here yet) is to use these learned motion estimates as pseudo-measurements in a filter (e.g., EKF) during simulated GPS outages.

### Raw dataset and sensors

- **Source**: EuRoC `MH_01_easy` ROS bag at `data/MH_01_easy.bag`
- **Topics used**
  - **IMU** `/imu0` → `sensor_msgs/Imu` (model input at inference time)
  - **Leica** `/leica/position` → `geometry_msgs/PointStamped` (ground truth used only to build labels)

Notes:
- Leica is *not* an input to the model; it is used to derive velocity labels for training/evaluation.

### High-level choices (so far)

- **Target**: predict \(\Delta v = v_{\text{end}} - v_{\text{start}}\) per window (3D)
- **Model**: Temporal Convolutional Network (TCN) baseline
- **Alignment**: interpolate Leica-derived velocity onto IMU timestamps
- **Splits**: chronological train/val/test split (to reduce leakage from overlapping windows)
- **Normalization**: compute per-channel mean/std from train only, apply to val/test
- **Debug workflow**: overfit a small subset before full training

Short decision writeups live in `docs/decisions/`.

### Preprocessing + training pipeline

All scripts use fixed paths under `data/` and write outputs under `data/processed/`.

Run in order:

```bash
python3 scripts/export_bag_topics.py
python3 scripts/derive_leica_velocity.py
python3 scripts/align_leica_to_imu.py
python3 scripts/build_training_windows.py
python3 scripts/split_and_normalize.py
python3 scripts/train_tcn_subset.py
```

#### Step 1 — Export bag topics to CSV

Script: `scripts/export_bag_topics.py`

Outputs:
- `data/processed/imu.csv`
- `data/processed/leica_position.csv`

#### Step 2 — Derive velocity from Leica position

Script: `scripts/derive_leica_velocity.py`

Method: first-order finite differences on position (baseline).

Output:
- `data/processed/leica_velocity.csv`

#### Step 3 — Align Leica velocity to IMU timeline

Script: `scripts/align_leica_to_imu.py`

Method: linear interpolation of Leica velocity onto IMU timestamps, keeping only IMU samples inside the Leica time range.

Output:
- `data/processed/imu_aligned_with_leica_velocity.csv` (IMU + `gt_vel_{x,y,z}`)

#### Step 4 — Build fixed-length training windows

Script: `scripts/build_training_windows.py`

Defaults:
- **Window size**: 200 samples (intended as ~1s at 200 Hz)
- **Stride**: 1

Saved outputs:
- `data/processed/X_windows.npy` (shape \((N, 200, 6)\))
- `data/processed/y_delta_v.npy` (shape \((N, 3)\))
- `data/processed/window_metadata.csv`

Feature order in `X_windows.npy`:
- `gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z`

#### Step 5 — Chronological split + train-only normalization

Script: `scripts/split_and_normalize.py`

Outputs:
- `data/processed/splits/{X,y}_{train,val,test}.npy`
- `data/processed/splits/normalization_stats.json` (ratios + feature order + train mean/std)

### Baseline model

- **Model code**: `src/models/tcn.py` (`TCNRegressor`)
- **Training sanity check**: `scripts/train_tcn_subset.py` (overfits a small subset to confirm the pipeline is wired correctly)

### Current status

Working end-to-end:
- export → label derivation → alignment → window building → splitting/normalization
- baseline TCN implementation
- tiny-subset overfit test (sanity check) succeeds

Not done yet:
- full training run + evaluation on the full dataset split
- saving checkpoints/metrics/plots in a repeatable way
- EKF/filter integration

### Next steps

- add a full training script (train + val, checkpoint best model)
- evaluate on test split (MSE/MAE, per-axis errors)
- add simple plots (learning curves, predicted vs target scatter, error histograms)
- iterate on labels if needed (smoothing / better differentiation), then revisit model architecture

### Decisions

- `docs/decisions/001-delta-velocity-target.md`
- `docs/decisions/002-export-bag-to-csv.md`
- `docs/decisions/003-derive-velocity-from-leica-position.md`
- `docs/decisions/004-use-finite-difference-for-first-velocity-derivation.md`
- `docs/decisions/005-use-interpolation-for-leica-to-imu-alignment.md`

### Repo layout

- `scripts/`: preprocessing + training entrypoints
- `src/`: model code
- `docs/decisions/`: short rationale notes
- `data/`: raw + processed artifacts (not all files are checked in)
