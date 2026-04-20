# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pipeline — run in order

All scripts must be run from the repo root so that relative `data/` paths resolve correctly.

```bash
python3 scripts/export_bag_topics.py        # ROS bag → CSVs
python3 scripts/derive_leica_velocity.py    # finite-difference velocity labels
python3 scripts/align_leica_to_imu.py       # interpolate labels onto IMU timestamps
python3 scripts/build_training_windows.py   # sliding windows → X_windows.npy, y_delta_v.npy
python3 scripts/split_and_normalize.py      # chronological split + train-only normalization
python3 scripts/train_tcn_subset.py         # sanity check: overfit small subset
python3 scripts/train_tcn_full.py           # full training run
python3 scripts/plot_loss_curves.py         # visualise results/tcn_baseline/loss_history.json
```

`export_bag_topics.py` requires ROS/rosbag Python bindings (Python 3.8 in the ROS env). All other scripts run under plain Python 3.

## Architecture

```
MH_01_easy.bag
  → imu.csv + leica_position.csv               (export_bag_topics)
  → leica_velocity.csv                          (derive_leica_velocity, finite diff)
  → imu_aligned_with_leica_velocity.csv         (align_leica_to_imu, linear interp)
  → X_windows.npy (N, 200, 6)                   (build_training_windows, stride=25)
    y_delta_v.npy (N, 3)
  → splits/{X,y}_{train,val,test}.npy           (split_and_normalize, chronological)
    normalization_stats.json
  → TCNRegressor (src/models/tcn.py)            (train_tcn_full)
  → checkpoints/tcn_improved.pt
    results/tcn_improved/{loss_history,test_metrics}.json
```

**Model** (`src/models/tcn.py`): `TCNRegressor` stacks `TemporalBlock` layers with exponentially growing dilation (`2^i`). Each block is two causal conv layers + residual. Input `(batch, time, 6)` is transposed to `(batch, 6, time)` before the network; the final time-step embedding is projected to Δv (3D).

Current config: `channel_sizes=[16,32,32]`, `kernel_size=3`, `dropout=0.3`, Adam lr=1e-3 + `weight_decay=1e-4`, early stopping patience=10, MSELoss.

**Window labelling**: `delta_v = vel[end] - vel[start]` where velocity is Leica-derived via finite differences, then linearly interpolated onto IMU timestamps. Leica is ground-truth only — never a model input.

**Split**: strictly chronological. Normalization stats (mean/std per channel) computed on train split only.

## Current status

| Run | Sequences | Train windows | Best val epoch | Val loss | Test MSE | Test MAE |
|---|---|---|---|---|---|---|
| Baseline (stride=1) | MH_01 only | ~180k | 1 | 0.04398 | 0.05012 | 0.16644 |
| Improved (stride=25) | MH_01 only | 1465 | 4 | 0.03842 | 0.04794 | 0.15809 |
| Multi-seq ⚠️ | MH_01–03+V1_01–02 | 5548 | 3 | 0.10596 | 0.36956 | 0.43399 |

⚠️ Multi-seq test is on MH_05_difficult (aggressive maneuvers) — NOT comparable to prior easy-sequence tests.
The higher MSE reflects harder flight dynamics, not pure regression. Y-axis error worst (mse_y=0.609).

**Next diagnostic**: IMU dead reckoning baseline on the same MH_05_difficult test set.
If dead reckoning MSE >> 0.37, the TCN is still useful for EKF integration.

Planned experiment progression (see `docs/experiments.md`):
1. IMU-only dead reckoning baseline
2. TCN motion prediction ← current
3. EKF with GPS
4. Neural-aided EKF (GPS-denied mode)

## Key numbers

| Parameter | Value |
|---|---|
| IMU rate | 200 Hz |
| Window size | 200 samples (~1 s) |
| Stride | 25 (1465 windows from MH_01_easy) |
| Input features | `gyro_{x,y,z}`, `accel_{x,y,z}` |
| Output | Δv (3D, m/s) |
| Train/val/test | chronological split |

## Organizational Memory

**Slug:** `gps-denied-nav`

- gbrain session notes → `sessions/YYYY-MM-DD/gps-denied-nav/`
- mem0 tag → `{"project": "gps-denied-nav"}`
- Linear team → GPS-Denied Nav
- Notion page → Projects/GPS-Denied Nav
- Obsidian folder → `gps-denied-nav/`

---

## Decisions

Short rationale notes live in `docs/decisions/`. Read them before changing the label pipeline or model target.
