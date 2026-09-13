# Decision 035: Hand-Derived Backward Kernel with Recompute-in-Backward

## Context
- Decision 034 shipped the forward kernel; end-to-end training through the filter also needs gradients w.r.t. state, covariance, IMU, the neural measurement `z`, its variance `r`, and the IMU noise `qc`.
- Autograd through the PyTorch reference stores every intermediate from ~20 ops per step (15×15 products, stacks, `where`), so a T-step rollout keeps tens of 15×15 tensors per filter per step.
- Options: (a) let autograd differentiate the reference; (b) save the full forward cache (~800 scalars per filter-step) and write a backward kernel; (c) save only the step inputs and recompute the forward intermediates inside the backward kernel.

## Decision
Use (c): `EKFStepFunction` saves only the step inputs (266 scalars per filter-step), and a hand-derived VJP kernel (`step_backward` in `ekf_math.h`) recomputes the forward pass per GPU thread before back-propagating through it.

## Reason
- Recomputing one step is cheap relative to the backward algebra and costs no extra launches, since it happens inside the same thread. It cuts saved memory ~3× compared with (b) and ~8× compared with autograd (a).
- All VJPs are closed-form and written out: Joseph-form covariance, adjugate 3×3 inverse, quaternion product and normalization, rotation matrix, and the Taylor-switched rotation-vector map, so the kernel has no dependence on autograd or LAPACK.
- The same header builds on CPU threads, so every gradient test runs on both backends.

## Consequences
- New: `gps_denied_nav/batched_ekf/function.py` (`EKFStepFunction`, `ekf_step_cuda`, a drop-in replacement for `reference.ekf_step`; broadcast `qc` gradients are summed by autograd), `tests/test_batched_ekf_backward.py`, `scripts/bench_batched_ekf_train.py`.
- Gradient parity with autograd through the reference (float64):
  - single step, CUDA and CPU: rtol 1e-8
  - 25-step rollout: rtol 1e-7
  - all-update and no-update branches
  - `gradcheck` on the extension itself
- Float32 kernel gradients are within ~1e-6 relative of float64, matching autograd's own float32 error. An early 144% "error" was a test bug: `randn` draws different loss weights for float32 than for float64.
- Training step (forward + backward over a 100-step rollout, velocity loss), RTX 4070 Laptop (`results/batched_ekf/bench_train_*.json`):

| dtype | N | autograd (reference) | custom kernel | speedup | peak memory |
|---|---:|---:|---:|---:|---|
| float32 | 1,024 | 42k steps/s | 879k | 20.8× | 816 → 100 MB |
| float32 | 8,192 | 267k | 2.48M | 9.3× | 6.6 GB → 799 MB |
| float64 | 1,024 | 54k | 351k | 6.5× | 1.7 GB → 206 MB |
| float64 | 4,096 | 126k | 316k | 2.5× | 6.6 GB → 799 MB |

- Float64 kernel throughput is flat from N = 1,024 to 4,096. The backward thread uses ~4,000 stack scalars and 10 dense 15×15 products, so the GPU is compute-saturated. Exploiting the sparsity of Φ and A = I − KH is the next optimization.
- `_ext.py` now builds with `MAX_JOBS=1` and removes build locks older than 15 min. On this 3.8 GB WSL host, a parallel nvcc build got OOM-killed and left a lock that hung every later `load()`.
