# Use `gps_denied_nav` on your own drone data

This tutorial walks through plugging your own dataset into the `NavPipeline`
API. End state: you can call `pipeline.run_outage(...)` on your data and get
a full velocity + position trajectory back, with whichever model and
adapter you prefer.

The package is intentionally composable — model, adapter, and filter are
independent. You can swap any one of them and keep the rest.

---

## 1. Install the package

```bash
git clone https://github.com/IshaanBansal2006/gps-denied-navigation
cd gps-denied-navigation
pip install -e .
```

Verify the imports:

```bash
python3 -c "from gps_denied_nav import NavPipeline; print('OK')"
```

---

## 2. The data contract

A "sequence" in this codebase is a fixed-rate IMU stream plus a
ground-truth velocity signal at the same rate. The reference loader
(`EuRoCSequence`) reads `imu_aligned.csv` with these columns:

```
timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,gt_vel_x,gt_vel_y,gt_vel_z
```

at 200 Hz, IMU in body frame, velocity in world frame.

If your data already looks like that, you can use `EuRoCSequence.load(...)`
directly — see §3.

If your data is in a different format, you need a class that exposes the
same four attributes as `EuRoCSequence`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class MyDroneSequence:
    name: str                # human-readable identifier
    timestamps: np.ndarray   # shape (N,)        — seconds since epoch (float64)
    imu: np.ndarray          # shape (N, 6)      — body-frame gyro_xyz + accel_xyz (float32)
    gt_vel: np.ndarray       # shape (N, 3)      — world-frame velocity (float32)

    @property
    def n_samples(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def outage_window(self, start_frac: float, duration_s: float):
        """Return (start_idx, end_idx) for an outage at start_frac of duration_s."""
        dt = self.timestamps - self.timestamps[0]
        start_t = self.duration_s * start_frac
        start_idx = int(np.searchsorted(dt, start_t))
        end_idx = min(self.n_samples - 1,
                       int(np.searchsorted(dt, start_t + duration_s)))
        return start_idx, end_idx
```

You don't need to subclass `EuRoCSequence` — `NavPipeline.run_outage` only
uses these four attributes + the `outage_window` method.

---

## 3. Load the data

For EuRoC data:

```python
from gps_denied_nav import EuRoCSequence
seq = EuRoCSequence.load("MH_05_difficult", "data/sequences")
print(f"{seq.name}: {seq.n_samples} samples, {seq.duration_s:.1f}s")
```

For your own data (using `MyDroneSequence` from §2):

```python
import pandas as pd, numpy as np
df = pd.read_csv("my_flight.csv")
seq = MyDroneSequence(
    name="flight_001",
    timestamps=df["timestamp"].to_numpy(dtype=np.float64),
    imu=df[["gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z"]].to_numpy(dtype=np.float32),
    gt_vel=df[["vel_x", "vel_y", "vel_z"]].to_numpy(dtype=np.float32),
)
```

---

## 4. Compose the pipeline

The three pieces are: a model that predicts velocity from IMU, an optional
adapter that specializes the model online from GPS-aided velocity, and a
filter that smooths the predictions.

```python
import torch
from gps_denied_nav import NavPipeline
from gps_denied_nav.models import load_lstm_checkpoint
from gps_denied_nav.adaptation import RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Model — load our pre-trained LSTM, or substitute your own.
model, norm = load_lstm_checkpoint("checkpoints/lstm_v15.pt", device)

# 2. Adapter — try None, RLSHead, or ContinuousAdapter.
adapter = RLSHead(in_dim=model.hidden_size, out_dim=3,
                   forgetting=0.995, p_init=0.1)

# 3. Filter — three-state velocity-only Kalman filter.
filt = VelocityOnlyFilter(sigma_process=0.5)

pipeline = NavPipeline(model=model, adapter=adapter, filter=filt,
                       norm=norm, device=device, update_stride=25)
```

---

## 5. Run a simulated GPS outage

```python
from gps_denied_nav import OutageEvaluator

evaluator = OutageEvaluator(seq, outage_start_frac=0.4)
result, metrics = evaluator.evaluate(pipeline, outage_duration_s=30.0)

print(f"Final velocity error: {metrics.final_velocity_error:.3f} m/s")
print(f"Final position drift: {metrics.final_position_drift:.2f} m")
```

`result` is an `OutageResult` dataclass with the full per-step
trajectory: `velocity_estimate`, `velocity_gt`, `position_estimate`,
`position_gt`, `timestamps`, plus convenience properties
`velocity_error`, `position_error`, `final_error_norm`, etc.

---

## 6. Swap components

The pipeline composes — swap any one piece, keep the rest.

**Different adapter:**

```python
from gps_denied_nav.adaptation import ContinuousAdapter
rls = RLSHead(in_dim=model.hidden_size, out_dim=3, forgetting=0.995, p_init=0.1)
cont = ContinuousAdapter(rls=rls, alpha_smooth=0.0,
                          ema_alpha=0.95, outage_lambda=1.0)
pipeline = NavPipeline(model=model, adapter=rls, filter=filt, norm=norm,
                       device=device, update_stride=25, continuous_adapter=cont)
```

**No adapter (raw model):**

```python
pipeline = NavPipeline(model=model, adapter=None, filter=filt,
                       norm=norm, device=device, update_stride=25)
```

**Your own filter** — write a class with `.reset(v0)`, `.predict(dt)`,
`.update(z)`, and `.velocity` property, then pass it in.

**Your own model** — write a PyTorch module that exposes `.lstm` (the
recurrent body) and `.head` (a linear projection), plus a `.hidden_size`
attribute. The pipeline reads features from `.lstm` and updates `.head` if
an adapter is configured. (For a different architecture entirely, you'd
need to fork `gps_denied_nav.pipeline.NavPipeline._step_features`.)

---

## 7. Train your own model

If you have data from a different drone class and want to fine-tune,
start from `scripts/train_lstm_v15.py` or `scripts/train_lstm_v18.py`.
The training loss is end-to-end navigation (a differentiable filter
rollout). You'll want to:

1. Replace `TRAIN_SEQS`, `VAL_SEQ`, `TEST_SEQ` with your sequence names.
2. Re-compute the IMU + velocity normalization stats on your training
   split — see `scripts/split_and_normalize.py` for the EuRoC version.
3. Adjust `WARMUP_LEN` / `OUTAGE_LEN` to match your operational scenario.

---

## What's not covered yet

- **No-GPS-truth fine-tuning**: there's no off-the-shelf training path
  for sequences where you only have IMU + (rough) ground-truth from
  another source like vision. Self-supervised training is a possible
  followup (see `gps_denied_nav/adaptation/continuous.py` for the
  inference-time version).
- **Real-time hardware deployment**: the pipeline runs in NumPy + PyTorch
  on CPU/GPU. No ONNX export or embedded-runtime conversion is shipped.

If you build either of those, a PR would be welcome.
