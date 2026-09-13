# Decision 034: CUDA Forward Kernel — One Thread per Filter, Shared Host/Device Math

## Context
- Decision 033 set up a batched PyTorch reference EKF; the reference on GPU runs ~20 small tensor kernels per step (15×15 matmuls, stacks, `where`), so at large N it's limited by kernel-launch overhead and memory traffic.
- The workload is embarrassingly parallel across filters but strongly sequential inside one step: 15×15 covariance algebra, quaternion math, a 3×3 inverse.
- A fair speedup claim needs a compiled CPU baseline, not just NumPy/Python.

## Decision
Implement the full step (predict + masked velocity update) as one CUDA kernel that assigns one GPU thread per filter and runs the whole step on fixed-size stack arrays, using one `__host__ __device__` header (`ekf_math.h`) that also builds a multithreaded C++ CPU path.

## Reason
- One kernel launch per step instead of ~20, and no intermediate device allocations: all temporaries (`StepCache`, ~800 scalars) live on the thread's stack.
- The shared header means the CPU and GPU paths are the same code, so parity tests cover both and the CPU benchmark is apples-to-apples.
- The 3×3 innovation covariance is inverted in closed form (adjugate), avoiding any batched LAPACK call.

## Consequences
- New: `gps_denied_nav/batched_ekf/csrc/{ekf_math.h, batched_ekf_cuda.cu, batched_ekf.cpp}`, `_ext.py` (JIT build via `torch.utils.cpp_extension.load`, arch list from the local GPU, actionable error if `nvcc` is missing), `scripts/bench_batched_ekf.py`, `tests/test_batched_ekf_cuda.py` (skipped in CPU-only CI).
- Kernel sources are shipped as package data in `pyproject.toml`.
- Parity: CUDA and C++ CPU match the reference over 200 steps × 64 filters (state rtol 1e-10, covariance rtol 1e-9, float64); float32 tracks float64 within 1e-3.
- Throughput, RTX 4070 Laptop, 40 steps, filter-steps/s (`results/batched_ekf/bench_forward_*.json`):

| Backend | N | float64 | float32 |
|---|---:|---:|---:|
| NumPy `EKF15` loop | 16 | 1.4k | (float64 only) |
| PyTorch reference, CPU | 16,384 | 111k | 185k |
| C++ same header, CPU threads | 65,536 | 232k | 305k |
| PyTorch reference, GPU | 65,536 | 533k | 1.53M |
| **Custom CUDA kernel** | 65,536 | **4.15M** | **9.79M** |

- At N = 65,536 the kernel is 7.8× (float64) and 6.4× (float32) faster than the same algorithm as PyTorch GPU ops, and 18× faster than compiled C++ on CPU threads. At N ≤ 64 the CPU C++ path wins, because per-step launch and sync cost dominates.
- NumPy timings on this laptop vary about 3× between runs (CPU power states), so the NumPy row is approximate.
- The dense 15×15 products ignore the sparsity of Φ and of A = I − KH; that optimization is deferred so the backward pass can be validated against a simple forward first.
