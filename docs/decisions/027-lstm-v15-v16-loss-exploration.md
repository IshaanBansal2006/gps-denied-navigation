# Decision 027: LSTM v15/v16 — Pure Nav Loss Wins, Hybrid Is a Wash

**Date:** 2026-05-21
**Status:** Accepted

## Context

Decision 025 identified end-to-end navigation loss as the most promising remaining
lever after per-step velocity MSE improvements saturated at v13. This decision
documents the two follow-on experiments:

- **v15** — pure nav loss: backprop through the 30-second rollout end-to-end, only
  the final position error feeds the gradient. No per-step velocity target.
- **v16** — hybrid: 0.5 × per-step velocity-weighted loss (v13 recipe) + 0.5 × nav
  loss over a 10-second rollout. Conjecture: per-step component anchors representation
  learning, nav component aligns the optimization with the real evaluation target.

Both trained on the remote RTX 2060 over 2026-05-19 → 2026-05-21 (~17 hours total).

## Results

| Model | Loss | nav_val_mean (m/s) | **nav_val_final (m/s)** | r2_mean | Best epoch |
|---|---|---|---|---|---|
| v13 baseline | per-step (vel-weighted) | 0.913 | 0.449 | **+0.207** | 14 |
| **v15** | **pure nav, 30s rollout** | 0.964 | **0.405** | -0.003 | 24 |
| v16 | hybrid 0.5 + 0.5, 10s rollout | 1.012 | 0.622 | -0.101 | 10 |

(`nav_val_final` is the validation rollout's velocity error at the final timestep —
the metric most correlated with downstream position drift at 30s.)

## Findings

### v15 is the new best for 30s final-position
- **0.405 m/s final velocity error** — first model to break below the 0.44–0.45 m/s
  floor that v7 / v11 / v13 were stuck at.
- Per-step accuracy collapsed (r²_mean from +0.207 to -0.003) — the model has
  *abandoned* per-step accuracy in favor of minimizing accumulated error at the rollout
  endpoint. That's exactly what we asked the loss to do, and it's the right tradeoff
  for the 30-second deployment scenario.
- Mean error over the rollout (0.964 m/s) is slightly worse than v13's 0.913 — the
  model gives up accuracy *during* the rollout to land closer to truth at the *end*.

### v16 hybrid is a wash
- Best epoch 10, then overfits — early stopped at 30.
- Worse than v13 on every per-step metric (r²_mean -0.101 vs +0.207).
- Worse than v15 on every nav metric (0.622 vs 0.405 final).
- The 10s nav window in v16 doesn't generalize to 30s evaluation; the per-step
  component prevents the model from specializing for end-of-rollout accuracy.
- Conclusion: pure nav (v15) beats hybrid (v16) on final-position; pure per-step (v13)
  beats hybrid on per-step accuracy. Hybrid loses on both fronts.

### Notable training dynamic (potential v17 follow-up)
v15's `val_final` kept dropping after the best-`val_mean` checkpoint was saved:

| Epoch | val_mean | val_final |
|---|---|---|
| 24 (selected) | **1.082** | 0.799 |
| 36 | 1.146 | 0.372 |
| 39 (early stop) | 1.147 | 0.374 |

The checkpoint-selection criterion (val_mean) saved the model at epoch 24 even
though val_final continued to improve to ~0.37 by epoch 39. If we re-ran v15 with
checkpoint selection on val_final, the headline number may improve another 5–10%.
This was deferred from this session.

## System recommendation (updated)

| Scenario | Best system |
|---|---|
| 5s GPS outage | LSTM v12 + VelFilter (0.171 m/s) |
| **30s final** | **LSTM v15 + VelFilter (0.405 m/s)** ⇐ new winner |
| 30s mean | LSTM v13 + VelFilter (0.913 m/s) |
| Simplest deploy | v7 TCN + VelFilter (stateless, no warmup) |

The portfolio narrative going forward: v15 is the headline. v13 is mentioned as the
best mean-error model. v7 is mentioned as the simplest deployable variant.

## What's left on the table

1. **v17: v15 with val_final checkpoint selection** — 1-line change, may unlock 5–10%.
2. **Sequence-level adaptation (RLS / TTT / LoRA)** — closes a different dimension of
   the gap to the GPS oracle (0.405 → 0.104 = 4× distance still open).
3. **Larger model with longer rollout curriculum** — unclear payoff, expensive.

(2) is the highest-impact direction but the highest-risk. (1) is the cheap consolation.
See decision 028 for why (2) is being deferred to focus on shipping the portfolio.
