# Decision 030: LSTM v18 — Bigger Model + Curriculum + val_final Selection

**Date:** 2026-05-24 (scaffold; numbers to be filled in when training completes)
**Status:** Pending — v18 training in progress on laptop RTX 4070; resumed from
epoch 25 (best val_final 0.9078). Expected to complete in 10–22 h from sprint
revision.

## Context

Decision 027 noted v15's `val_final` continued to drop well past the
`val_mean` minimum that the script used for checkpoint selection. The
training-dynamic figure showed `val_final` reaching ~0.37 by epoch 36, vs
the saved epoch-24 checkpoint's 0.80. Up to 5–10 % of the headline number
was potentially left on the table.

v18 tests three changes simultaneously:

1. **Bigger model.** Hidden 128→256, layers 2→3, dropout 0.3→0.4. Goes
   from ~202k params to ~1.3M.
2. **Curriculum training.** OUTAGE_LEN grows over phases: 10 s (epochs
   1-15) → 20 s (epochs 16-30) → 30 s (epochs 31-60). Lets the larger
   network converge on the long-horizon target without diverging early.
3. **`val_final` checkpoint selection.** Replaces `val_mean` selection
   from decision 027's flagged limitation.

## Approach

`scripts/train_lstm_v18.py` — fork of v15 trainer with the three changes
above. LR also reduced (5e-5 → 3e-5) for the bigger network, PATIENCE
extended (15 → 20) to absorb curriculum-transition spikes.

**RESUME=1** env var: if the training process is interrupted (e.g., GPU
preempted for another task), the checkpoint already on disk preserves the
optimizer + scheduler state. Resumption picks up exactly where it left
off. Used during this sprint when the laptop GPU was momentarily freed
for the user's other work.

Evaluation (`scripts/neural_aided_ekf_v18.py`):
- Vanilla v18 + velocity-only filter
- v18 + RLS (same hyperparameters as decision 029)
- v18 + continuous adapter (same hyperparameters as decision 032)
- All at 5 / 10 / 30 / 60 s outage durations on MH_05.

## Results — TO BE FILLED IN

| System | 5s | 10s | **30s** | 60s |
|---|---:|---:|---:|---:|
| Vanilla v15 + filter (reference) | 0.505 | 1.089 | 0.403 | 0.803 |
| Vanilla v18 + filter | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| v18 + RLS | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| v18 + continuous | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

(Final velocity error at outage end, m/s.)

## Findings — TO BE FILLED IN

### Did v18 outperform v15?
_TBD: yes/no/depends on horizon._

### Did curriculum help?
_TBD: the training dynamic across the three OUTAGE_LEN phases — did each
transition produce a val_final regression, did the network recover?_

### Was val_final the right selection criterion?
_TBD: the saved best-epoch from val_final vs what val_mean would have picked._

## Decision — TO BE FILLED IN

_TBD: keep v15 as production, keep v18 as alternate, swap, or punt._

## What this changes about the project

- New checkpoint: `checkpoints/lstm_v18.pt`.
- New train script: `scripts/train_lstm_v18.py` (with `RESUME=1` support).
- New eval script: `scripts/neural_aided_ekf_v18.py`.
- If v18 wins headline: update `README.md` + `docs/figures/baseline_comparison.{png,svg}` +
  `docs/figures/trajectory_animation.gif` + `CLAUDE.md` results table.
- If v18 doesn't win: this doc is the honest write-up of why.

## What's left

If v18 ties or regresses: cross-dataset eval (TUM-VI / KITTI) is the next
attack — bigger model has more capacity to specialize to new distributions
once given exposure.
