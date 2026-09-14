# Decision 037: Learn EKF Noise by Backprop Through the CUDA Kernels — Improves the EKF, Doesn't Beat the Velocity-Only Filter

## Context
- Decisions 033–036 built a differentiable batched 15-state EKF whose training step is 16× faster than autograd. This is its first use on real data.
- Decision 019 hand-swept one strapdown-EKF knob (σ_tcn) and found the velocity-only filter better during outages. The question here is whether gradient descent over all noise parameters changes that verdict.
- Protocol (`scripts/learn_ekf_noise.py`, `gps_denied_nav/batched_ekf/rollout.py`):
  - Windows: 10 s GPS-aided warmup (ground-truth velocity every sample, σ = 0.02, as in decisions 015/019), then a 30 s outage with LSTM v15 velocity every 25 samples.
  - Initial attitude, including yaw, comes from a Kabsch alignment of gyro-propagated specific force to ground-truth velocity increments.
  - Learned (log-parameterized, Adam): 4 IMU noise groups `qc` and 3 per-axis LSTM measurement σ. Initialized at decision 019's best hand-tuned EKF.
  - Loss: mean squared velocity error over the outage.
  - Data: train on windows from the 6 train sequences, select on MH_04, report on MH_05. Every system is evaluated on identical windows and identical LSTM measurements.
- Before training, the script checks kernel vs autograd-reference gradients on real data: 5e-13 relative.

## Decision
Keep the velocity-only filter in the headline pipeline. Ship the learned-noise EKF tooling and results as an ablation, not as a replacement.

## Reason
- Learning helps the EKF, and the gain reproduces across two runs and two GPUs. Mean outage error on test drops 8–9 % from the hand-tuned setting, and final error drops slightly.
- It still loses to the velocity-only filter on test mean final error (1.060 vs 1.051) and clearly on mean outage error (1.117 vs 0.924). Validation agrees.
- The learned parameters point the same way as decision 019. Accel noise σ rose 0.05 → 0.084 (trust IMU propagation less), and LSTM measurement σ fell 0.01 → 0.003–0.007 (trust the network more). The optimizer is pushing the EKF toward a velocity-only filter.

## Consequences
- Results (MH_05 test, 30-s outages, final and mean velocity error in m/s):

| Run | Windows (train / eval) | Velocity-only filter | EKF hand-tuned | EKF learned |
|---|---|---|---|---|
| RTX 4070 Laptop, 40 iters (`learned_noise.json`) | 48 / 20 | 1.037 / **0.905** | 1.063 / 1.229 | 1.052 / 1.127 |
| RTX 2060, 60 iters (`learned_noise_2060_large.json`) | 192 / 40 | **1.051 / 0.924** | 1.075 / 1.229 | 1.060 / 1.117 |

- Random-window numbers are higher than the 0.403 single-window headline. At the headline window the velocity-filter evaluation here reproduces 0.4031 exactly (checked against `NavPipeline`), so the protocol matches.
- Open issue, not caused by the kernels: after warmup the EKF's bias estimates are physically implausible (|b_a| ≈ 6–12 m/s², |b_g| up to 2 rad/s) in every variant tried: 8 Hz or 200 Hz GPS, 10 s or 30 s warmup, default or tight bias priors. This points to unmodeled IMU–Leica time offset, lever arm, or extrinsics in the derived velocity labels. It caps any strapdown-EKF result on this data, including end-to-end LSTM training through the filter, until it's fixed.
- Throughput in practice: 192 × 30-s windows (1.54M filter-steps forward, 1.15M backward) take 3.2 s per iteration on an RTX 2060.
- Desktop notes: the toolchain script and all 40 tests pass on the RTX 2060 (compute capability 7.5, torch 2.5.1, Python 3.10). Runs there must set `TMPDIR` off `/tmp` and run in the foreground of a held SSH session, because a detached run lost its compiler temp files when the WSL session closed.
- New: `gps_denied_nav/batched_ekf/rollout.py`, `scripts/learn_ekf_noise.py`, `tests/test_batched_ekf_rollout.py`, `scripts/make_cuda_ekf_figure.py`, `docs/figures/cuda_ekf_benchmark.{png,svg}`, and a README section.
