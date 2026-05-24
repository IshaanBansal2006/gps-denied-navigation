# Decision 032: Continuous Adaptation During Outage — Val/Test Disagree

**Date:** 2026-05-24
**Status:** Accepted (with val/test caveat — read §"What changed late")

## Context

Decision 029 noted that RLS adaptation freezes the head the moment the GPS
outage starts. The natural extension is to keep updating the head *during*
the outage using a self-supervised signal — no ground truth is available,
but physics constraints (gyro-integration consistency) and self-distillation
(pulling toward the filter's smoothed output) can provide pseudo-targets.

## Approach

`gps_denied_nav/adaptation/continuous.py` — `ContinuousAdapter` wraps the
existing `RLSHead`. During the outage, at every STRIDE-th sample:

1. **Smoothed-filter pseudo-target.** EMA the filter's velocity estimate; use
   the EMA as a soft "predict more like recent history" signal.
2. **Gyro-rotated previous-velocity pseudo-target.** Rodrigues-rotate the
   previous filter velocity by `omega * dt` from the gyro. If motion is
   purely rotational this is exactly correct; with translation, it's a
   biased anchor.
3. Convex-combine into a single pseudo-target with weight ``alpha_smooth``.
4. RLS update with a softer forgetting factor (``outage_lambda`` ≈ 0.999
   vs 0.995 in pre-outage) because pseudo-targets are less reliable than GT.

## Sweep on val MH_04_difficult (30 s outage at start_frac=0.4)

| Pseudo-target mix | final_vel | final_pos |
|---|---:|---:|
| Vanilla v15 + filter (baseline) | **0.7820** | 17.15 |
| v15 + filter + RLS only (baseline) | **0.7495** | 26.00 |
| Continuous α=1.0 (pure smoothing) | 0.9039 | 26.12 |
| Continuous α=0.7 (mixed) | 0.8796 | 26.01 |
| Continuous α=0.3 (mixed) | 0.8458 | 25.88 |
| Continuous α=0.0 (pure gyro) | 0.8192 | 25.81 |
| Continuous α=0.0, λ=0.9999 (weaker updates) | 0.8109 | 25.84 |
| Continuous α=0.0, λ=1.0 (no forgetting) | 0.8100 | 25.84 |
| Continuous α=0.5, λ=1.0 | 0.8465 | 25.94 |

## Findings

### Continuous adaptation alone is worse than RLS alone
- Best continuous: 0.8100 final_vel (α=0, λ=1.0).
- v15 + RLS alone:  0.7495 final_vel.
- Continuous costs 8 % on the headline metric.

### Pure gyro pseudo-target is the strongest of the variants
- α=0 (pure Rodrigues-rotated previous velocity) consistently wins.
- The smoothed-filter EMA pseudo-target is essentially circular: pulling
  predictions toward the EMA of predictions is a regularizer toward
  *constancy*, not toward truth.

### Why continuous adaptation doesn't help here
The pre-outage RLS warmup already converged the head to a sequence-specific
linear fit. During the outage, the only new information is the IMU stream
itself, which the LSTM already consumes. The self-supervised pseudo-targets
don't add a genuinely new signal:
- Filter smoothing is a property of the filter, not new information.
- Gyro consistency assumes negligible translational acceleration, which is
  violated during the swooping motion at frac=0.6 on MH_05 (the headline
  scenario).

For continuous adaptation to win, the pseudo-target needs to encode physics
the model *hasn't already captured*. Candidates for future work:

1. **Zero-velocity update (ZUPT)** if motion can be detected as stationary
   (variance of recent IMU below threshold). EuRoC has no such windows; a
   drone landing/hovering dataset would.
2. **Magnetometer fusion** for absolute orientation anchoring — not in EuRoC.
3. **Visual odometry pseudo-velocity** from monocular flow — the EuRoC rig
   has cameras; integration deferred.

## What changed late

After committing the negative-on-val finding above, the best-by-val
continuous config (α=0, λ=1.0) was also run on **test MH_05** for the
ablation table. Surprise:

| System | val MH_04 final_vel | **test MH_05 final_vel** |
|---|---:|---:|
| v15 + filter (vanilla)  | 0.782 | 0.403 |
| v15 + filter + RLS      | **0.750** ← val best | 0.259 |
| v15 + filter + continuous (α=0) | 0.819 | **0.246** ← test best |
| v15 + filter + TTT+RLS  | 0.764 | 0.258 |

Continuous adaptation **lost on val** but **won on test** by 5 %. This is a
val/test methodology conflict: the val sequence (MH_04) and test sequence
(MH_05) have different motion characteristics, and method ranking flipped.

## Decision

Two options for headline reporting:

1. **Strict val-selection**: report only the val-selected winner. v15+RLS at
   0.259 stays as the headline; continuous's 0.246 is mentioned but not the
   featured number.
2. **Inclusive ablation**: report all configs' test numbers transparently;
   note that continuous wasn't val-selected. Reader sees both numbers and can
   judge.

**Chose option 2** (inclusive ablation). The README ablation table includes
all four configs with their test numbers; the prose explicitly flags that
continuous wasn't selected on val and the headline number conservatively
stays at the RLS value 0.259. Posting 0.246 as the headline would imply
val-test consistency that this experiment doesn't have.

The honest takeaway for the recruiter: "this person ran the experiment,
saw a val/test conflict, refused to overclaim, documented both numbers."
That's a stronger signal than a 5 % single-shot improvement on the test set.

## What this changes about the project

- New module: `gps_denied_nav/adaptation/continuous.py` (~120 lines).
- New eval script: `scripts/neural_aided_ekf_lstm_v15_continuous.py`.
- `NavPipeline` extended with optional `continuous_adapter=` parameter.
- README ablation table now includes Continuous (0.246) and TTT+RLS (0.258)
  as additional attempts, with the val/test caveat.
- Headline stays at 0.259 (val-selected RLS).
