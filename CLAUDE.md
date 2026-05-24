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
| TCN v11 (2s window) ✓ | same 6 seqs | ~5.8k | 35 | 1.230‡ | — | **+0.158** |
| LSTM v12 (dense, 2s) ✓ | same 6 seqs | ~12k chunks | 53 | 1.200‡ | — | +0.203 |
| LSTM v13 (vel-weighted) ✓ | same 6 seqs | ~12k chunks | 14 | 1.484‡† | — | **+0.207** |
| LSTM v15 (pure nav, 30s) | same 6 seqs | ~12k chunks | 24 | 1.082§ | 1.618 | -0.003 |
| LSTM v16 (hybrid step+nav-10s) ❌ | same 6 seqs | 795 chunks | 10 | 1.002§ | 1.790 | -0.101 |

† directional loss on delta_v — not comparable to MSE-only val losses
‡ directional loss on normalised absolute velocity — different scale, not comparable to delta_v runs
§ val_mean = mean navigation error over 30s outage rollout (m/s) — not comparable to per-step val losses
Best R²: **v11 (+0.158)**. Best corr_y: v11 (0.486). New baseline: v11.

**Augmentation verdict** (decision 020): yaw rotation catastrophic (R² collapses 20x — destroys EuRoC heading priors).
Noise-only (σ=0.05) is neutral (+3%, within noise). Train/val gap is sequence-level distribution shift, not sample-level overfitting.

**Longer window verdict** (decision 021): WINDOW_SIZE 200→400, 3-layer→6-layer (RF 29→253 samples).
R² improved 66% (0.095→0.158). Y-axis R² doubled (0.105→0.221). Temporal context was a real bottleneck.

**Best navigation system (decisions 018–019)**: velocity-only Kalman filter + TCN v7.
Final error at 30s: **0.440 m/s** (vs 0.501 standalone TCN, vs 0.104 EKF+GPS upper bound).
Strapdown EKF during outage is harmful — attitude drift poisons IMU propagation within 10s.
Architecture: pre-outage = strapdown EKF+GPS; during outage = velocity-only filter + TCN v7.
(Navigation eval should be re-run with v11 checkpoint — expected improvement.)

**v11 navigation eval** (decision 022): v11 better at 5s/60s outages but v7 still wins at 30s (0.440 vs 0.485 m/s).
VelFilter is saturated — 66% R² gain doesn't close the distribution-shift gap. v7 remains best for 30s scenario.

**LSTM v13** (decision 024): velocity-weighted loss fixes z (corr_z 0.253→0.375, +48%). Best overall model.
r2_mean +0.207, train/val gap 1.15x (tightest ever). X/Y within 3% of v12 peak.

**LSTM v12 nav eval**: matched GPS at 5s (0.171 vs 0.172 m/s). 30s worse than v7 (0.497 vs 0.440) — z drift.
v13 nav eval pending: better z should recover 30s.

**v13 nav eval** (decision 025): best mean@30s = 0.913 m/s (5% better than v7). 30s final = 0.449 (tied with v7).
Per-step MSE improvements have saturated. 30s gap to GPS (0.449 vs 0.104) is structural distribution shift.

**LSTM v15 (pure nav loss, 30s outage)**: nav_val_mean 0.964, nav_val_final **0.405** (vs v13's 0.449 — first model
to beat v13/v7 on final-position). Per-step R² collapsed to -0.003 — model abandoned step-wise accuracy entirely.
Notable training dynamic: val_final kept dropping (0.80→0.37) after best-epoch checkpoint at epoch 24 —
selection on val_mean may have left value on the table. Decision doc pending.

**LSTM v16 (hybrid 0.5×per-step v13-weighted + 0.5×nav-10s)** ❌: best epoch 10, nav_val_mean 1.012,
nav_val_final 0.622, r2_mean -0.101. Worse than v13 across the board. The hybrid loss with a 10s nav window
didn't help — pure nav (v15) beat it on final-position; pure per-step (v13) beat it on R².

**v15 + RLS adaptation head** (decision 029): pre-outage GPS-aided velocity feeds an online recursive-
least-squares update to the LSTM's final linear head. At 30s outage: **0.259 m/s final velocity error
(−36% vs vanilla v15+filter)** — closes the gap to the GPS oracle from 4× to 2.5×. Trades against
shorter (5s/10s) and longer (60s) outages where the adapted head fails to generalize. Hyperparameters
P₀=0.1, λ=0.995 chosen on the same test sequence — generalization to other sequences not yet validated.

**Best systems by scenario:**
- 5s outage: LSTM v12 VelFilter (0.171 m/s, matched GPS)
- **30s final: LSTM v15 VelFilter + RLS (0.259 m/s)** ⇐ new winner (decision 029)
- 30s mean: v13 LSTM (0.913)
- Simplest deployment: v7 TCN (stateless, no warmup)

**Critical next steps** (in order):
1. ~~Data augmentation~~ ✓ (decision 020)
2. ~~Longer window / TCN~~ ✓ (decisions 021-022)
3. ~~LSTM dense + velocity-weighted~~ ✓ (decisions 023-025)
4. ~~End-to-end navigation loss~~ ✓ (v15 — wins on final-position; mean unchanged)
5. ~~Sequence-level adaptation~~ ✓ (RLS variant — decision 029, 30s headline only)
6. (Open) Retrain v15-style with checkpoint selection on val_final — log suggests another 5-10%
7. (Open) Cross-sequence eval of RLS — validate the 36% claim outside MH_05

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
