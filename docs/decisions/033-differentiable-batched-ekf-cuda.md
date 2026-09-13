# Decision 033: Differentiable Batched EKF as Custom CUDA Kernels

## Context
- Goal: a portfolio piece that shows real CUDA skill to both ML-systems (PI, OpenAI, Anthropic) and autonomy/onboard (Anduril, Boston Dynamics, Skydio) engineers.
- Options considered: A) custom TCN dilated-conv kernel, B) fused LSTM cell, C) batched EKF kernel, D) Triton kernels, E) differentiable batched EKF (C + hand-written backward).
- A/B compete with cuDNN and likely lose; D hides the CUDA-level details; C shows no autograd integration.
- The existing `EKF15` (`gps_denied_nav/filters/ekf.py`) is NumPy on CPU and steps one filter at a time; Monte Carlo evaluation loops filters sequentially.
- The user waived the `~/projects/` scaffolding-only rule for this feature and asked for full implementation with internal testing, pushed per iteration.

## Decision
Build option E: a batched 15-state error-state EKF step (predict + masked velocity update) as a C++/CUDA PyTorch extension with a hand-derived backward kernel, validated against a pure-PyTorch batched reference.

## Reason
- The only option that is strong for both audiences: custom forward/backward kernels + `autograd.Function` integration, and GPU-parallel state estimation with fused 15×15 covariance algebra.
- Fixes a real bottleneck (one CPU filter at a time) and enables end-to-end training through the filter (Haarnoja et al. 2016, Backprop KF).
- A pure-PyTorch reference gives an exact oracle: forward parity with `EKF15`, and autograd gradients the CUDA backward must match.

## Consequences
- New package `gps_denied_nav/batched_ekf/`:
  - `reference.py` — batched, differentiable PyTorch EKF step (packed `x (N,16)`, `P (N,15,15)`, `imu`, `z`, `r`, `qc`, `mask`).
  - `state.py` — packing to/from `EKF15` states.
- Rotation-vector → quaternion uses a Taylor branch below θ = 0.05 rad so the map and its derivative stay smooth at θ = 0 and precise in float32 (the NumPy version hard-switches to identity below 1e-10).
- `scripts/setup_cuda_toolchain.sh` installs a sudo-free CUDA 12.1 `nvcc` (NVIDIA conda channel) matching torch cu121; pip `nvidia-cuda-nvcc-cu12` wheels do not ship the `nvcc` binary.
- Iteration 1 results: reference matches `EKF15` over a 300-step, 6-filter random trajectory (state rtol 1e-9, covariance rtol 1e-8) and passes `torch.autograd.gradcheck` w.r.t. all six inputs.
- Next iterations: CUDA forward kernel + parity/benchmark, hand-derived backward kernel + gradient parity, fused multi-step rollout, end-to-end training demo.
