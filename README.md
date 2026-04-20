> ⚠️ **Work in Progress** — Active research project. Pipeline, model architecture, and results are evolving. See `docs/experiments.md` for the current state.

---

## Neural-Aided GPS-Denied Navigation for UAVs

Build a navigation system that keeps a UAV accurately positioned when GPS is unavailable — jammed, spoofed, indoors, or in contested airspace. Inspired by Shield.AI's Hivemind principle: zero external dependencies.

**Approach**: IMU windows → neural network (TCN) predicts delta velocity → EKF fuses predictions as pseudo-measurements during GPS outages.

---

### Pipeline

Run from repo root in order:

```bash
# Step 1 — Process each EuRoC sequence (requires ROS Python for bag export)
python3 scripts/process_sequence.py data/raw/MH_01_easy.bag MH_01_easy
python3 scripts/process_sequence.py data/raw/MH_02_easy.bag MH_02_easy
python3 scripts/process_sequence.py data/raw/MH_03_medium.bag MH_03_medium
python3 scripts/process_sequence.py data/raw/V1_01_easy.bag V1_01_easy
python3 scripts/process_sequence.py data/raw/MH_04_difficult.bag MH_04_difficult
python3 scripts/process_sequence.py data/raw/V1_02_medium.bag V1_02_medium

# Step 2 — Assemble dataset (plain Python 3)
python3 scripts/build_dataset.py

# Step 3 — Train
python3 scripts/train_tcn_full.py

# Step 4 — Visualise
python3 scripts/plot_loss_curves.py
```

> `process_sequence.py` skips steps if outputs already exist — safe to re-run.

---

### Architecture

```
EuRoC bags (MH_01–04, V1_01–02)
  → per-sequence: imu.csv, leica_velocity.csv, imu_aligned.csv   (process_sequence.py)
  → per-sequence: X_windows.npy (N, 200, 6), y_delta_v.npy (N, 3)
  → data/splits/{X,y}_{train,val,test}.npy                       (build_dataset.py)
  → TCNRegressor  →  Δv (3D)                                     (train_tcn_full.py)
```

**Model** (`src/models/tcn.py`): Stacked causal dilated conv blocks with residual connections. Input `(batch, 200, 6)` → final time-step embedding → linear head → Δv (3D).

**Labels**: `delta_v = vel[end] - vel[start]` using Leica/Vicon-derived velocity. Ground truth is never a model input at inference.

**Split** (sequence-based, no leakage):
| Split | Sequences | Purpose |
|---|---|---|
| Train | MH_01, MH_02, MH_03, V1_01 | Mixed difficulty + environments |
| Val | MH_04 | Harder MH dynamics |
| Test | V1_02 | Cross-environment generalization |

---

### Current Results

| Run | Config | Best val epoch | Val loss | Test MSE | Test MAE |
|---|---|---|---|---|---|
| Baseline | stride=1, [32,64,64], dropout=0.1 | 1 | 0.04398 | 0.05012 | 0.16644 |
| Improved | stride=25, [16,32,32], dropout=0.3 | 4 | 0.03842 | 0.04794 | 0.15809 |
| Multi-seq | stride=25, [32,64,64], dropout=0.2 | TBD | TBD | TBD | TBD |

---

### Roadmap

- [x] Single-sequence TCN baseline
- [x] Stride/regularization improvements
- [x] Multi-sequence pipeline
- [ ] Multi-sequence training (in progress)
- [ ] IMU-only dead reckoning baseline
- [ ] EKF (velocity-state, from scratch)
- [ ] Neural-aided EKF with simulated GPS outages
- [ ] Evaluation suite (drift vs. outage duration)
- [ ] ONNX export for hardware deployment

---

### Key Numbers

| Parameter | Value |
|---|---|
| IMU rate | 200 Hz |
| Window | 200 samples = 1 s |
| Stride | 25 |
| Input | gyro_x/y/z, accel_x/y/z |
| Output | Δv (3D, m/s) |

### Decisions

Short rationale notes live in `docs/decisions/` (001–010). Read before changing the label pipeline, model target, or split strategy.
