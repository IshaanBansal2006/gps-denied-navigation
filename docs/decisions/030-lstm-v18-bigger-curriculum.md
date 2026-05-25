# Decision 030: LSTM v18 — Bigger Model + Curriculum + val_final Selection

**Date:** 2026-05-25
**Status:** Accepted (negative result — v18 ties or marginally regresses vs v15)

## Context

Decision 027 noted v15's `val_final` continued to drop well past the
`val_mean` minimum used for checkpoint selection. The training-dynamic
figure showed `val_final` reaching ~0.37 by epoch 36, vs the saved
epoch-24 checkpoint's 0.80. Up to 5–10 % of the headline number was
potentially left on the table.

v18 tested three changes simultaneously:

1. **Bigger model.** Hidden 128→256, layers 2→3, dropout 0.3→0.4. Goes
   from ~202 k params to ~1.3 M.
2. **Curriculum training.** OUTAGE_LEN grows over phases: 10 s
   (epochs 1-15) → 20 s (epochs 16-30) → 30 s (epochs 31-60). Lets the
   larger network converge on the long-horizon target without diverging
   early.
3. **`val_final` checkpoint selection.** Replaces `val_mean` selection
   from decision 027's flagged limitation.

## Training summary

- Trained over 2026-05-24 → 2026-05-25 on laptop RTX 4070 Laptop.
- 60 epochs total. Process was killed once at epoch 25 and resumed
  cleanly via `RESUME=1` (decision-030-side benefit: the train script
  now supports interrupt-resume).
- Best checkpoint: **epoch 56**, `val_final` = 0.8980 m/s.

## Results — vanilla v18 + filter (no adapter), MH_05 30 s outage

| Model | Best `val_final` (MH_04) | Test `nav_val_final` (MH_05) | Test `nav_val_mean` |
|---|---:|---:|---:|
| v15 (decision 027) | 0.405 (epoch 36, never selected) | **0.403** | 0.964 |
| v15 *selected* (epoch 24)  | 0.799 (selected on val_mean) | 0.403 | 0.964 |
| **v18** | **0.898** | **0.430** | 0.981 |

### Key finding: v18 underperforms v15 even on its own selection criterion

- v18 chose `val_final` selection explicitly. Its best `val_final` is
  **0.898** — much worse than v15's eventual val_final minimum at epoch 36
  (~0.37).
- On the test sequence, v18's 30 s final velocity error is **0.430 m/s**,
  vs v15's 0.403 m/s. A 7% regression.
- The mean test velocity error follows the same pattern (0.981 vs 0.964).

### Per-axis correlations and R²
v18: r2_mean=0.024, corr_x=0.21, corr_y=0.34, corr_z=-0.02.
v15: r2_mean=-0.003 (decision 027). v18 has marginally better per-step R²
but worse end-of-rollout (the metric that matters).

## Why it didn't work

Two plausible explanations, both supported by the val curves:

1. **Underfit despite 4× more parameters.** v18's val_final plateaued
   around 0.90 from the first 30 s curriculum phase (epoch 31) onwards.
   v15's val_final at the same epoch range was already in the 0.4–0.6 band.
   The bigger network apparently could not match v15's representational
   convergence within 30 epochs of 30 s-rollout training.
2. **Curriculum was the wrong structural intervention.** The 10 s → 20 s
   → 30 s schedule means the model trains on the harder objective for
   only ~30 epochs total (vs v15's 50 epochs on the full 30 s objective).
   v15's "throw 30 s at it from epoch 1" approach actually gave more
   gradient steps on the production objective.

This is not the dominant explanation though — v18's parameters were also
~6.5× more, so even fewer 30 s gradient steps × more parameters means
each step had less specific information per parameter.

## What this changes about the project

- New checkpoint: `checkpoints/lstm_v18.pt` (epoch 56, val_final 0.898).
  Kept for the multi-duration ablation below but not used in production.
- `checkpoints/lstm_v15.pt` remains the production model.
- README headline: stays at v15 + filter + RLS = **0.259 m/s**.
- New train script: `scripts/train_lstm_v18.py` (with `RESUME=1` support).
- New eval script: `scripts/neural_aided_ekf_v18.py`.

## Adapter combinations on v18 (pending desktop eval — fill in)

| System | 5s | 10s | 30s | 60s |
|---|---:|---:|---:|---:|
| Vanilla v18 + filter | _TBD_ | _TBD_ | **0.430** | _TBD_ |
| v18 + filter + RLS | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| v18 + filter + continuous | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

(30 s vanilla number from above is the training script's
`nav_val_final_err` on the test set.)

A bigger model that can't match the smaller model on its own metric is
unlikely to suddenly win with an adapter on top, but the runs are cheap
and the data is recruiter-visible — running them as part of the desktop
queue.

## What's left

If v18 + adapter doesn't shift the picture: **v18 ships as a documented
attempt, not a production model.** The deferred queue stays the same
(cross-dataset eval, ZUPT, vision fusion). The lesson — that bigger model
+ curriculum does not automatically beat a smaller model trained on the
production objective end-to-end — is the contribution.

If v18 + an adapter unexpectedly wins: update README headline. (Not
expected; the size of v18's val_final gap to v15 is too large for a
6 % adapter delta to close.)
