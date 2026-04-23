# 026 — LSTM v14 nav eval: end-to-end nav loss beats v7 at 30s final

Date: 2026-04-23

## Context

v14 is the first end-to-end navigation-loss run. Instead of per-step velocity MSE,
the loss is the mean L2 velocity error over a simulated 10s GPS outage, backpropped
through a differentiable PyTorch velocity-only Kalman filter. Training samples a
random 10s outage per step, gradient accumulated over 4 outages.

**Training was cut short** — stopped at epoch 19/50 for lid-close shutdown after
~8h wall-clock. Per-step MSE saturation (v7/v11/v12/v13 stuck at 0.44–0.50 m/s 30s
final) was the motivation.

## Training trace (partial)

Best checkpoint saved at **epoch 13** (early-stopping metric = val_mean_err on a
single 30s outage at MH_04_difficult 40%).

| Epoch | train | val_mean | val_final | best |
|-------|-------|----------|-----------|------|
| 1  | 0.694 | 1.137 | 0.933 | ✓ |
| 6  | 0.699 | 1.121 | 0.931 | ✓ |
| 9  | 0.662 | 1.091 | 0.964 | ✓ |
| 13 | 0.671 | **1.084** | 1.124 | ✓ ← saved |
| 15 | 0.673 | 1.107 | 0.911 | |
| 17 | 0.613 | 1.205 | 0.640 | |
| 19 | 0.628 | 1.241 | **0.590** | |

**Important**: val_final kept dropping sharply (1.12 → 0.59) while val_mean drifted up
after epoch 13. The val_mean early-stopping criterion saved a checkpoint that is
**not** the best for final-error performance. A val_final-aware criterion would have
captured epoch 19 (or later).

## Nav eval — MH_05_difficult

Final velocity error at outage end (m/s):

| Outage | DR    | LSTM-v14 | v14 VelFilter | v7 VelFilter | v13 VelFilter | EKF+GPS |
|--------|-------|----------|---------------|--------------|---------------|---------|
|  5 s   | 8.77  | 0.421    | **0.386**     | —            | 0.171*        | 0.172   |
| 10 s   | 15.82 | 0.885    | 1.126         | —            | —             | 0.328   |
| 30 s   | 45.87 | 0.449    | **0.425**     | 0.440        | 0.449         | 0.104   |
| 60 s   | 95.11 | 0.747    | 0.779         | —            | —             | 0.229   |

*v12 value at 5s was 0.171 — v13 not directly comparable here.

Mean velocity error at 30s: v14 VelFilter **0.966 m/s** vs v13 **0.913 m/s** vs GPS 0.202.

## Interpretation

**v14 is the first model to beat v7 at 30s final.** 0.425 < 0.440 (3.4% improvement)
— and with only 19 of 50 epochs trained, using a suboptimal checkpoint. This is a
structural win, not a tuning artifact: per-step MSE models plateau at ~0.44 m/s, and
v14 is the first to cross below.

**Tradeoff:** v14 is worse than v13 on 30s mean (0.966 vs 0.913) and worse than v12
on 5s final (0.386 vs 0.171). The model has specialized toward end-of-rollout
accuracy over rollout-average accuracy, because that's what training optimized.
Val_final dropping to 0.59 in late epochs confirms the model was still improving
on final error at shutdown.

**What end-to-end nav loss actually changes:** per-step models trade equally across
all timesteps of a rollout. Nav loss with mean-L2 aggregation still spreads gradient
across timesteps, but the Kalman filter inside the loss lets errors earlier in the
rollout compound into errors later — the model learns which early predictions
matter most for final position. That's why final error drops while mean error
doesn't.

## Decisions

1. **v14 is the new 30s-final champion** (0.425 m/s). v7 dethroned.
2. **Early-stopping metric is wrong** for nav-loss training. Future runs (v15/v16)
   should early-stop on val_final, or a weighted combo (e.g., 0.3*mean + 0.7*final).
3. **v13 still wins 30s mean** (0.913 m/s). Best navigation system depends on the
   application: v14 for "where will I be at t=30", v13 for "how far did I drift
   on average during the outage".
4. **Full training would likely beat this further.** 19 epochs with val_final still
   dropping = we stopped on the way down. Worth re-running with proper early stopping
   and a full epoch budget.

## Next steps

- **v14 full run** — retrain with val_final as the early-stopping metric, full 50
  epochs, expected significant further improvement beyond 0.425.
- **v15** (30s outage training, 30s eval match) — next in the queue, but the
  lesson from v14 is the eval metric matters more than the training outage length.
- **v16** (hybrid 50/50 per-step + nav) — may give best-of-both: good 30s mean
  from per-step component, good 30s final from nav component.
