# GPS-Denied Navigation for UAVs

> Keep a drone navigated when GPS dies — 30 seconds of GPS-free flight at **0.259 m/s final velocity error** on EuRoC MH_05, 2.5× the GPS-aided EKF oracle, 170× better than naive IMU dead-reckoning.

![Hero — 30-second outage on MH_05](docs/figures/hero.png)

The blue line is a neural-aided IMU navigator built from scratch in this repo, ending **3.13 m from ground truth** after a simulated 30-second GPS outage on the held-out EuRoC MH_05_difficult test sequence. The grey dotted line is a GPS-aided EKF oracle (the ceiling). The red dashed line is what would happen if you trusted the IMU alone.

![30-second animation](docs/figures/trajectory_animation.gif)

---

## Why this exists

GPS goes down all the time — urban canyons, jammers, indoor flights, contested airspace. Most autonomous UAVs become unusable within seconds of losing GPS because raw IMU integration drifts catastrophically (the red line above hits ~620 m of drift in 30 seconds).

The standard fix is an EKF that fuses IMU with whatever else you can get. This project asks: **what if the "whatever else" is a neural network that learned what IMU motion looks like on similar drones?**

Built as a portfolio piece on the path to robotics research; the goal was an honest, end-to-end implementation of the full pipeline — data, model, filter, evaluation — with documented decisions at every step.

---

## Headline result

**Final velocity error after a 30-second simulated GPS outage on EuRoC MH_05_difficult (the held-out test sequence):**

![Baseline comparison](docs/figures/baseline_comparison.png)

| System | 30-second final velocity error | × oracle |
|---|---:|---:|
| IMU dead-reckon (no model) | 45.867 m/s | 441× |
| TCN v7 + velocity-only filter | 0.440 m/s | 4.2× |
| LSTM v13 + velocity-only filter | 0.449 m/s | 4.3× |
| LSTM v15 + velocity-only filter | 0.403 m/s | 3.9× |
| **LSTM v15 + filter + RLS adaptation** | **0.259 m/s** | **2.5×** |
| EKF + GPS (oracle ceiling) | 0.104 m/s | 1.0× |

The RLS adaptation head is the headline contribution — see [decision 029](docs/decisions/029-rls-adaptation-head.md) for the honest write-up including the trade-offs at shorter horizons.

---

## Quickstart (5 minutes)

```bash
# 1. Clone
git clone https://github.com/IshaanBansal2006/gps-denied-navigation
cd gps-denied-navigation

# 2. Install
pip install -r requirements.txt        # torch, numpy, pandas, matplotlib

# 3. Reproduce the headline (requires EuRoC MH_05_difficult — see data/README.md)
python3 scripts/neural_aided_ekf_lstm_v15_rls.py --outages 30
# Final → 0.259 m/s velocity error after 30 s GPS outage
```

To regenerate the hero figure: `python3 scripts/make_hero_figure.py`
To regenerate the animation: `python3 scripts/make_trajectory_animation.py`
To regenerate the comparison plot: `python3 scripts/make_baseline_comparison_figure.py`

---

## Architecture

```
EuRoC .bag  (6 training sequences, MH_04 val, MH_05 test)
   │
   ├── export_bag_topics.py    →  imu.csv + leica_position.csv
   ├── derive_leica_velocity.py →  finite-diff velocity labels
   ├── align_leica_to_imu.py   →  imu_aligned.csv  (200 Hz, 6 IMU channels + 3 vel)
   ├── build_training_windows.py
   └── split_and_normalize.py  →  X_{train,val,test}.npy   y_{train,val,test}.npy

                ┌────────────────────────────────────┐
   IMU @ 200 Hz │ LSTM v15 (128 hidden, 2 layers)    │  ← frozen at deploy
   ────────────▶│  └ Linear head 128→3 (vel)         │  ← adapts online (RLS)
                └──────────────┬─────────────────────┘
                                │ velocity prediction (every 25th sample)
                                ▼
                       ┌──────────────────┐
                       │  velocity-only   │  P, Q, R per-channel
                       │     filter       │  fuses LSTM into a smooth velocity estimate
                       └────────┬─────────┘
                                ▼
                          Δposition  →  GPS-free navigation
```

**Pre-outage:** the LSTM runs forward to warm its hidden state; the RLS head ingests GPS-aided velocity targets at every 25th sample (~358 updates over 90 s of warmup) to specialize to the current sequence.

**During outage:** the LSTM body stays frozen, the (now-adapted) head predicts velocity from the hidden state, the velocity-only filter smooths the predictions.

Model code: [`src/models/`](src/models/), filter: [`src/filters/ekf.py`](src/filters/ekf.py), RLS head: [`src/adaptation/rls.py`](src/adaptation/rls.py).

---

## Approach — the decision trail

This project ran 16 model variants, 7 nav-eval studies, and 29 numbered decision docs. The high-leverage moves, in chronological order:

| Decision | What changed | Why it mattered |
|---|---|---|
| [015](docs/decisions/015-neural-aided-ekf-gps-denied-mode.md) | Built the neural-aided EKF, defined the GPS-denied evaluation protocol | Established the evaluation discipline |
| [017](docs/decisions/017-tcn-v7-absolute-velocity-target.md) | Switched TCN from Δv → absolute v target | First model to break +0.09 R²; unblocked everything downstream |
| [019](docs/decisions/019-velocity-only-filter-beats-strapdown-ekf.md) | Replaced the strapdown EKF with a velocity-only filter | Attitude drift was poisoning IMU propagation within 10 s |
| [021](docs/decisions/021-tcn-v11-longer-window.md) | Window 1 s → 2 s, 3 conv layers → 6, RF 29 → 253 samples | R² jumped +66 %; temporal context was the bottleneck |
| [023](docs/decisions/023-lstm-v12-sequence-model.md) | Migrated TCN → dense LSTM | r²_mean from +0.158 → +0.203 |
| [024](docs/decisions/024-lstm-v13-velocity-weighted-loss.md) | Velocity-weighted loss | Fixed Z-axis: corr_z 0.253 → 0.375 (+48 %) |
| [027](docs/decisions/027-lstm-v15-v16-loss-exploration.md) | End-to-end navigation loss over 30 s rollouts (v15) | First model to break 0.45 m/s on 30 s final-position |
| [029](docs/decisions/029-rls-adaptation-head.md) | **RLS adaptation head — closed 30 s final-err 0.403 → 0.259** | Headline. Closes gap to oracle from 4× to 2.5× |

Each decision doc contains the hypothesis, the result, and what was learned — including the negative results (decisions [020](docs/decisions/020-augmentation-findings.md) augmentation; [027](docs/decisions/027-lstm-v15-v16-loss-exploration.md) v16 hybrid loss; [029](docs/decisions/029-rls-adaptation-head.md) RLS at non-30s horizons).

---

## What it took — training dynamic for v15

![v15 training dynamic](docs/figures/loss_curves_v15.png)

End-to-end nav loss is noisy (red line keeps falling even after `val_mean` plateaus). The selected checkpoint at epoch 24 may have left 5–10 % on the table — a follow-up retrain with `val_final` selection is queued.

---

## What's honest about this

- **All numbers are on EuRoC MH_05_difficult**, the held-out test sequence. The model and the RLS hyperparameters were tuned on the same sequence; cross-dataset evaluation is queued (decision 029 §Limitations).
- The RLS adaptation head **helps at the 30 s outage horizon and hurts everywhere else** (5 s, 10 s, 60 s). It is not a general-purpose drop-in replacement — it specializes to the training horizon.
- The dead-reckoning baseline integrates body-frame acceleration *without* gravity compensation, which is the project's documented "naïve" baseline. A full strapdown INS without GPS would be less dramatic but still bad (decision 019).
- All training is on the EuRoC dataset (TUM stereo+IMU rig, ~1 km of total flight). Generalization to other drones, other IMUs, and outdoor flight is unknown.

---

## What's next

The deferred-experiments queue, in priority order ([decision 029](docs/decisions/029-rls-adaptation-head.md) §What's left):

1. **Cross-sequence eval of the RLS head** — does the 36 % win on MH_05 hold up on TUM-VI or KITTI?
2. **v17 — v15 retrained with `val_final` checkpoint selection** — 1-line change, plausible 5–10 % more
3. **Continuous adaptation during outage** using a self-supervised pretext task (e.g., gyro integration consistency)
4. **LoRA-style adaptation** — low-rank adapters on the LSTM body, not just the head

---

## Layout

```
gps-denied-navigation/
├── src/
│   ├── models/         (TCN, LSTM regressors)
│   ├── filters/        (15-state EKF, velocity-only filter)
│   └── adaptation/     (RLSHead — online linear adaptation)
├── scripts/            (pipeline, training, nav-eval, figure scripts)
├── data/sequences/     (per-sequence imu_aligned.csv + windowed numpy)
├── checkpoints/        (trained model binaries — v7, v11, v12, v13, v15)
├── results/            (per-model nav-eval JSONs, loss histories)
└── docs/
    ├── decisions/      (29 dated decision docs)
    ├── figures/        (hero, baseline, loss curves, trajectory GIF)
    └── plan-portfolio-ship.md
```

---

## Stack

Python 3.8+ · PyTorch 2.4 (CUDA 12.1) · NumPy · Pandas · Matplotlib · EuRoC MAV dataset

Trained on a single RTX 2060. Pipeline runs end-to-end on a laptop CPU.

---

## License

MIT
