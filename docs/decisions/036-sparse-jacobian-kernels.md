# Decision 036: Exploit Jacobian Sparsity Instead of Dense 15×15 Products

## Context
- Decisions 034/035 used dense 15×15 products for Φ P Φᵀ, the Joseph update A P Aᵀ, and their VJPs: about 10 dense products per backward thread, each 3,375 multiply-adds.
- Float64 training throughput stopped scaling at N ≈ 1k–4k, which means each GPU thread was compute-bound.
- Structure available: F has five nonzero 3×3 blocks, and three of them are constant (I, −I, and zero rows 9–14). A = I − KH differs from I only in columns 3:6.

## Decision
Never materialize Φ or A. Apply F through four block operators (`f_x`, `x_ft`, `ft_x`, `x_f`) and write the Joseph update as rank-3 corrections, in both the forward and the hand-derived backward kernels.

## Reason
- Φ P Φᵀ = E + dt·E Fᵀ with E = P + dt·F P. Each F product touches 9 rows (or columns) × 15 with 3×3 inner loops, instead of 15×15×15.
- A P1 = P1 − K·P1[3:6,:] and (A P1)Aᵀ = A P1 − (A P1)[:,3:6]·Kᵀ, so the update never forms a 15×15×15 product. In the backward pass, only columns 3:6 of ∂L/∂A are needed, because they are the only ones that reach K.
- Per-thread stack use drops too: the cache no longer holds Φ or A. That matters for GPU kernel threads.

## Consequences
- `gps_denied_nav/batched_ekf/csrc/ekf_math.h` only; Python API unchanged. All parity and gradient tests pass unchanged (rtol 1e-8 vs autograd, including `gradcheck`).
- The benchmark scripts gained `--repeats` (median) and `--backends`. An earlier single-shot run was too noisy on this laptop GPU (e.g. float64 training at N=4,096 swung 4× between runs), so numbers now come from a same-session A/B that rebuilds each header variant.
- A/B, RTX 4070 Laptop (`results/batched_ekf/bench_ab_*_{dense,sparse}_*.json`), filter-steps/s:

| Path | dtype | N | dense | sparse | speedup |
|---|---|---:|---:|---:|---:|
| CUDA forward | float32 | 65,536 | 10.5M | 18.4M | 1.75× |
| CUDA forward | float64 | 65,536 | 3.72M | 6.16M | 1.66× |
| CUDA train (fwd+bwd, 100 steps) | float32 | 8,192 | 2.54M | 4.18M | 1.64× |
| CUDA train (fwd+bwd, 100 steps) | float64 | 4,096 | 0.79M | 1.59M | 2.00× |
| C++ CPU threads forward | float32 | 64 | 0.58M | 2.38M | 4.1× |

- Cumulative for float32 training at N = 8,192, against autograd through the PyTorch reference: 240k → 4.18M filter-steps/s (17×), with peak memory 6.6 GB → 799 MB.
- The forward kernel at N = 16,384 is repeatably slower than at N = 1,024 in both variants. This is a laptop/WSL effect not yet explained, so it's reported as measured rather than smoothed over.
