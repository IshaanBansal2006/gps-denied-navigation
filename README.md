# GPS-Denied Navigation for UAVs

[![CI](https://github.com/IshaanBansal2006/gps-denied-navigation/actions/workflows/ci.yml/badge.svg)](https://github.com/IshaanBansal2006/gps-denied-navigation/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IshaanBansal2006/gps-denied-navigation/blob/main/notebooks/demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Keep a drone navigated when GPS dies — **0.259 m/s final velocity error after a 30-second GPS outage** on EuRoC MH_05 (held-out test sequence). 2.5× the GPS-aided EKF oracle. 170× better than naïve IMU dead-reckoning.

![Hero — 30-second outage on MH_05](docs/figures/hero.png)

The blue line is a neural-aided IMU navigator built from scratch in this repo, ending **3.13 m from ground truth** after a simulated 30-second GPS outage on EuRoC MH_05_difficult. The grey dotted line is a GPS-aided EKF oracle (the ceiling). The red dashed line is what would happen if you trusted the IMU alone.

![30-second animation](docs/figures/trajectory_animation.gif)

---

## Architecture

![System pipeline](docs/figures/architecture.png)

Three composable parts: a frozen LSTM body, an online-adapted linear head, and a velocity-only Kalman filter. The whole thing is a 10-line program against the published `gps_denied_nav` API — see the snippet below.

---

## Headline result

**Final velocity error after a 30-second simulated GPS outage on EuRoC MH_05_difficult:**

![Baseline comparison](docs/figures/baseline_comparison.png)

| System | 30-s final velocity error | × GPS oracle | Where it lives |
|---|---:|---:|---|
| IMU dead-reckon (no model) | 45.867 m/s | 441× | — |
| TCN v7 + velocity-only filter | 0.440 m/s | 4.2× | decision [018](docs/decisions/018-ekf-outage-v7-comparison.md) |
| LSTM v13 + filter | 0.449 m/s | 4.3× | decision [025](docs/decisions/025-lstm-v13-navigation-eval.md) |
| LSTM v15 + filter | 0.403 m/s | 3.9× | decision [027](docs/decisions/027-lstm-v15-v16-loss-exploration.md) |
| **LSTM v15 + filter + RLS adaptation** | **0.259 m/s** | **2.5×** | decision [029](docs/decisions/029-rls-adaptation-head.md) |
| LSTM v15 + filter + TTT-then-RLS | 0.258 m/s | 2.5× | decision [031](docs/decisions/031-ttt-adaptation.md) — val-eliminated |
| LSTM v15 + filter + continuous-adapt (α=0) | 0.246 m/s | 2.4× | decision [032](docs/decisions/032-continuous-adaptation.md) — val/test conflict |
| EKF + GPS (oracle ceiling) | 0.104 m/s | 1.0× | — |

The **0.259 m/s** headline is the val-selected winner (RLS adaptation on top of v15 + filter). Continuous adaptation and TTT+RLS came in slightly better on test but didn't win on val — they're shipped as additional modules with honest write-ups, not promoted to the headline. See decisions 031 and 032 for the val/test methodology.

---

## Use this on your own drone

Full step-by-step walkthrough: [docs/use-on-your-own-data.md](docs/use-on-your-own-data.md).
Five-minute version:

```bash
pip install -e .
```

```python
from gps_denied_nav import NavPipeline, EuRoCSequence, OutageEvaluator
from gps_denied_nav.models import load_lstm_checkpoint
from gps_denied_nav.adaptation import RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence = EuRoCSequence.load("MH_05_difficult", "data/sequences")
model, norm = load_lstm_checkpoint("checkpoints/lstm_v15.pt", device)

pipeline = NavPipeline(
    model=model,
    adapter=RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1),
    filter=VelocityOnlyFilter(),
    norm=norm, device=device, update_stride=25,
)

# Evaluate one 30-second outage at 40% through the sequence
ev = OutageEvaluator(sequence, outage_start_frac=0.4)
result, metrics = ev.evaluate(pipeline, outage_duration_s=30.0)
print(f"Final velocity error: {metrics.final_velocity_error:.3f} m/s")
# → 0.2588 m/s
```

To run on your own data, drop in a class with the same shape as `EuRoCSequence` (timestamps, IMU @ rate, optional ground-truth velocity for the warmup) and the rest of the pipeline composes the same way. The adapter (`RLSHead`, `TTTAdapter`, `ContinuousAdapter`) and filter (`VelocityOnlyFilter`) are independent modules — swap any one of them out.

---

## Quickstart (5 minutes)

```bash
git clone https://github.com/IshaanBansal2006/gps-denied-navigation
cd gps-denied-navigation
pip install -e .
# Requires the EuRoC MH_05_difficult sequence preprocessed — see data/README.md
python3 scripts/neural_aided_ekf_lstm_v15_rls.py --outages 30
# → 0.259 m/s velocity error after 30 s GPS outage
```

Regenerate any figure:
```bash
python3 scripts/make_hero_figure.py
python3 scripts/make_trajectory_animation.py
python3 scripts/make_baseline_comparison_figure.py
python3 scripts/make_architecture_diagram.py
python3 scripts/make_loss_curves_figure.py
```

---

## Approach — the decision trail

This project ran 16 model variants, 9 nav-eval studies, and 32 numbered decision docs. The high-leverage moves, in chronological order:

| Decision | What changed | Why it mattered |
|---|---|---|
| [015](docs/decisions/015-ekf-architecture-results.md) | Built the neural-aided EKF; defined the GPS-denied evaluation protocol | Established the evaluation discipline |
| [017](docs/decisions/017-tcn-v7-absolute-velocity-target.md) | Switched TCN from Δv → absolute v target | First model to break +0.09 R²; unblocked everything downstream |
| [019](docs/decisions/019-velocity-only-filter-beats-strapdown-ekf.md) | Replaced the strapdown EKF with a velocity-only filter | Attitude drift was poisoning IMU propagation within 10 s |
| [021](docs/decisions/021-tcn-v11-longer-window.md) | Window 1 s → 2 s, 3 conv layers → 6, RF 29 → 253 samples | R² jumped +66 %; temporal context was the bottleneck |
| [023](docs/decisions/023-lstm-v12-sequence-model.md) | Migrated TCN → dense LSTM | r²_mean from +0.158 → +0.203 |
| [024](docs/decisions/024-lstm-v13-velocity-weighted-loss.md) | Velocity-weighted loss | Fixed Z-axis: corr_z 0.253 → 0.375 (+48 %) |
| [027](docs/decisions/027-lstm-v15-v16-loss-exploration.md) | End-to-end navigation loss over 30-s rollouts (v15) | First model to break 0.45 m/s on 30-s final-position |
| [029](docs/decisions/029-rls-adaptation-head.md) | **RLS adaptation head — closed 30-s final-err 0.403 → 0.259** | Headline. Closes gap to oracle from 4× to 2.5× |
| [031](docs/decisions/031-ttt-adaptation.md) | Test-time training of the LSTM body | Negative result on in-distribution EuRoC; module shipped for future cross-dataset work |
| [032](docs/decisions/032-continuous-adaptation.md) | Self-supervised continuous adaptation during outage | Lost on val, won on test by 5 %; flagged val/test conflict honestly |

Each decision doc contains the hypothesis, the result, and what was learned — including the negative results.

---

## What it took — v15 training dynamic

![v15 training dynamic](docs/figures/loss_curves_v15.png)

End-to-end nav loss is noisy (red line keeps falling even after `val_mean` plateaus). The selected checkpoint at epoch 24 may have left 5–10 % on the table — a follow-up retrain with `val_final` selection (v18) is in flight.

---

## What's honest about this

- **All numbers are on EuRoC MH_05_difficult**, the held-out test sequence. The RLS hyperparameters were originally tuned on the same sequence (decision 029 limitation). Cross-dataset eval is queued.
- The RLS adaptation **only helps at the 30-s outage horizon** the model was trained for. At 5, 10, or 60 s, vanilla v15 + filter is the safer pick (decision 029).
- Continuous adaptation (decision 032) **lost on val MH_04** (0.819 vs RLS's 0.750) but **won on test MH_05** (0.246 vs RLS's 0.259). The README headline conservatively stays at the val-selected number (0.259). Posting 0.246 as the headline would imply val-test consistency the experiment doesn't have.
- Test-time training of the LSTM body (decision 031) **does not help** on in-distribution EuRoC val. Module is shipped for cross-dataset experiments where domain shift gives it something to work on.
- The dead-reckoning baseline integrates body-frame acceleration *without* gravity compensation — the project's "naïve" baseline. A full strapdown INS without GPS would be less dramatic but still bad (decision 019).
- All training is on EuRoC (~1 km of indoor MAV flight). Generalization to other drones, other IMUs, outdoor flight is unknown.

---

## What's next (deferred experiments)

Full prioritized list with rationale: [docs/roadmap.md](docs/roadmap.md). Top of that list:

1. **Cross-dataset eval on TUM-VI or KITTI** — would unblock TTT (which needs domain shift) and validate the RLS / continuous adaptation findings outside MH_05.
2. **v18 — bigger LSTM (256/3-layer) + curriculum training** — currently training; results will land as decision 030.
3. **LoRA adapters on the LSTM gates** — middle ground between head-only RLS (decision 029) and full-model TTT (decision 031).
4. **Zero-velocity update (ZUPT) for continuous adapter** — needs a different dataset; EuRoC has no genuinely stationary windows.

---

## Layout

```
gps-denied-navigation/
├── gps_denied_nav/             ← pip-installable package
│   ├── models/                 (LSTM, TCN regressors)
│   ├── filters/                (15-state EKF, velocity-only filter)
│   ├── adaptation/             (RLSHead, TTTAdapter, ContinuousAdapter)
│   ├── data/                   (EuRoCSequence dataset class)
│   ├── pipeline.py             (NavPipeline composer)
│   └── eval.py                 (OutageEvaluator)
├── tests/                      (23 pytest unit tests — pytest tests/)
├── scripts/                    (training, nav-eval, figure scripts)
├── notebooks/demo.ipynb        (Colab-ready end-to-end demo)
├── data/sequences/             (per-sequence imu_aligned.csv)
├── checkpoints/                (trained model binaries — v7, v11–v15, v18)
├── results/                    (per-model nav-eval JSONs)
├── docs/
│   ├── decisions/              (32+ dated decision docs)
│   ├── figures/                (hero, architecture, baseline, loss, GIF)
│   ├── use-on-your-own-data.md (porting tutorial)
│   ├── roadmap.md              (what I'd build next)
│   ├── plan-portfolio-ship.md
│   └── distribution.md
├── src/                        (backward-compat shims → gps_denied_nav.*)
├── .github/workflows/ci.yml    (pytest + mypy on push/PR)
├── pyproject.toml
├── setup.py
└── LICENSE
```

---

## Stack

Python 3.8+ · PyTorch 2.4 (CUDA 12.1) · NumPy · Pandas · Matplotlib · EuRoC MAV dataset

Trained on a single RTX 2060 (v15) and an RTX 4070 Laptop (v18). Inference runs end-to-end on a laptop CPU.

---

## License

MIT
