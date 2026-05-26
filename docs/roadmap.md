# Roadmap

What I'd build next if I came back to this project. Ordered by expected
return per unit effort, drawn from the open items across the decision docs.

---

## Near-term (1–2 weeks of work each)

### Cross-dataset evaluation on TUM-VI or KITTI

- **Why:** every claim in this repo is on EuRoC, where val and test
  sequences come from the same MAV rig in the same lab. The 0.259 m/s
  headline + the RLS adaptation win + the TTT "wash" all rely on the
  test data being in-distribution. The next credible test is a different
  drone, different IMU, different environment.
- **What it tests:** does v15 + filter + RLS still beat vanilla v15 +
  filter when the test data is genuinely OOD? Does TTT (decision 031)
  finally help when the model sees something it wasn't trained on?
- **Cost:** ~1 week to wire up the new dataset's preprocessing + run
  evals.

### Retrain v15-style with `val_final` checkpoint selection (v17)

- **Why:** decision 027 noted v15's `val_final` continued dropping for
  ~15 epochs past the saved val_mean-best checkpoint. v18 (decision 030
  pending) bundles this with a bigger model and curriculum, conflating
  three variables. A clean v17 with *only* the selection criterion change
  would isolate the effect.
- **Cost:** one line of code, one training run (~17 h on RTX 2060).

### LoRA adapters on LSTM gates

- **Why:** RLS only adapts the final linear head; TTT adapts the whole
  body but overfits the warmup window. LoRA is the middle ground —
  low-rank deltas on the LSTM weight matrices, trained on the pre-outage
  window. Smaller adaptation capacity than TTT, less forgetting risk.
- **Why deferred:** predictably low value on in-distribution EuRoC (same
  failure mode as TTT, per decision 031). The win venue is the
  cross-dataset eval above — wait for that infrastructure first.
- **Cost:** ~1 week (LoRA on `nn.LSTM` gates is fiddly; need to override
  the forward pass).

---

## Medium-term (1+ months)

### Zero-velocity update (ZUPT) for the continuous adapter

- **Why:** decision 032's continuous adapter uses gyro-rotated previous
  velocity + EMA smoothing as pseudo-targets. The strongest available
  self-supervised signal is *zero-velocity detection*: when IMU variance
  drops below threshold, the drone is hovering or landed and the velocity
  is exactly zero — a hard pseudo-target.
- **Why not now:** EuRoC has no genuinely stationary windows; the dataset
  was filmed during continuous flight. A dataset with explicit hovering
  segments (or pre/post-flight stationary periods) would unlock this.

### Visual-inertial fusion via the existing `NavPipeline`

- **Why:** the EuRoC rig has stereo cameras. Visual odometry (e.g.
  SVO/ORB-SLAM) could feed pseudo-velocity into the same filter as a
  parallel measurement stream. This is where modern UAV navigation
  actually lives — IMU + vision + (when available) GPS.
- **Cost:** substantial — wiring up a visual-odometry frontend + the
  measurement model in the filter. ~1 month. Would live in a new
  `gps_denied_nav/sensors/visual.py` module.

### Online learning during operation

- **Why:** continuous adaptation (decision 032) uses self-supervised
  signals during the outage. Online learning *while GPS is available*
  would extend that: every GPS update is a training signal for the
  LSTM body, not just the head. Risk: forgetting the prior distribution.
- **Why deferred:** open research question, not a clean engineering win.

---

## Long-term / aspirational

### Hardware deployment on a real drone

- Take the trained `lstm_v15.pt` + `NavPipeline`, export the LSTM to ONNX
  or TFLite, run it on a Pixhawk or NVIDIA Jetson companion computer
  during a real GPS-denied flight.
- The package is structured to make this feasible (the inference path
  is pure NumPy + PyTorch eval-mode), but the surface area to actually
  pull it off is large.

### Different vehicle classes

- Trained on a 700g quadcopter (EuRoC) with one specific IMU. A fixed-wing
  aircraft or a ground rover has fundamentally different dynamics — would
  the architecture transfer with a re-trained model, or does it need
  vehicle-class-specific tweaks (e.g., different filter Q, different
  WARMUP_LEN, different adaptation hyperparameters)?

---

## Methodology / non-modeling work

### Statistical rigor passes

- Cross-sequence RLS validation — does the 36% win on MH_05 replicate
  on MH_03 / V1_03 / MH_04?
- Monte Carlo distribution of the headline — is the single-point
  0.259 m/s robust or a lucky window?
- Multi-duration sweep — full operating curve (5–60 s outages) for each
  system, not just the 30-s headline.

### A real "papers I'm trying to imitate" section

- Decisions 015–029 reference the right ideas (TCN, LSTM, RLS, TTT,
  continuous adaptation, curriculum learning) but cross-references to
  the actual originating papers are thin. A `docs/related-work.md`
  with proper citations would help anyone trying to extend this.

### CI matrix

- Current `.github/workflows/ci.yml` runs pytest + mypy on Python 3.10
  and 3.11. Could add 3.12, lint with ruff, build the package, run a
  documentation-link check.

---

## Out of scope (deliberately not doing)

- **Reinventing the EKF.** The 15-state EKF in `gps_denied_nav.filters.ekf`
  works fine for the pre-outage warmup. Re-deriving with different state
  parameterizations (e.g. error-state Kalman, IEKF) won't move the
  headline.
- **Real-time C++ port.** Out of scope for a portfolio piece. The Python
  inference path runs ~real-time on a laptop CPU already.
- **Web-based interactive demo.** Colab notebook (`notebooks/demo.ipynb`)
  is the demo. A bespoke web app would be cool but uses time better
  spent on the items above.
