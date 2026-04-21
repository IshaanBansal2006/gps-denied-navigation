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
| Multi-seq (corrected) | MH_01–03+V1_01–02 train / MH_04 val / MH_05 test | 5548 | 11 | 0.10577 | 0.08914 | 0.21136 |
| TCN v2 ❌ | same split | 5548 | 1 | 0.40043† | 0.09264 | -0.013 |
| TCN v3 | same split | 5548 | 27 | 0.41265† | 0.09086 | +0.003 |
| TCN v4 (SG labels) | same split | 5548 | 38 | 0.40278† | 0.08511 | -0.001 |
| TCN v5 (EKF labels) | same split | 5548 | 18 | 0.42109† | 0.09006 | +0.004 |
| TCN v6 (+V1_03) ✓ | +V1_03_difficult | 6385 | 44 | 0.40938† | 0.08978 | **+0.013** |
| TCN v7 (abs vel) ✓ | same 6 seqs | 6385 | 72 | 1.287‡ | 1.465‡ | **+0.095** |
| TCN v8 (large) | same 6 seqs | 6385 | 19 | 1.300‡ | — | +0.099 |
| TCN v9 (aug: rot+noise) ❌ | same 6 seqs | 6385×aug | 44 | 1.497‡ | — | +0.005 |
| TCN v10 (aug: noise only) | same 6 seqs | 6385×aug | 76 | 1.283‡ | — | +0.098 |

† directional loss on delta_v — not comparable to MSE-only val losses
‡ directional loss on normalised absolute velocity — different scale, not comparable to delta_v runs
Best R²: v10 (+0.098, statistically tied with v7). Best checkpoint: **v7** (consistent, noise augmentation is neutral).

**Augmentation verdict** (decision 020): yaw rotation catastrophic (R² collapses 20x — destroys EuRoC heading priors).
Noise-only (σ=0.05) is neutral (+3%, within noise). Train/val gap is sequence-level distribution shift, not sample-level overfitting.

**Best navigation system (decisions 018–019)**: velocity-only Kalman filter + TCN v7.
Final error at 30s: **0.440 m/s** (vs 0.501 standalone TCN, vs 0.104 EKF+GPS upper bound).
Strapdown EKF during outage is harmful — attitude drift poisons IMU propagation within 10s.
Architecture: pre-outage = strapdown EKF+GPS; during outage = velocity-only filter + TCN v7.

**Critical next steps** (in order):
1. ~~Data augmentation~~ ✓ DONE — noise-only neutral, rotation catastrophic (decision 020)
2. Longer window (400 samples / 2s) — more IMU context per prediction → TCN v11
3. Different architecture (LSTM/Transformer) — state across windows
4. End-to-end navigation loss — train directly on EKF drift

Planned experiment progression (see `docs/experiments.md`):
1. IMU-only dead reckoning baseline ✓
2. TCN motion prediction ✓ (multi-seq)
3. EKF with GPS ✓
4. Neural-aided EKF (GPS-denied mode) ✓ (decision 015)

## Key numbers

| Parameter | Value |
|---|---|
| IMU rate | 200 Hz |
| Window size | 200 samples (~1 s) |
| Stride | 25 (1465 windows from MH_01_easy) |
| Input features | `gyro_{x,y,z}`, `accel_{x,y,z}` |
| Output | absolute velocity (3D, m/s) — z-score normalised in v7+ |
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
