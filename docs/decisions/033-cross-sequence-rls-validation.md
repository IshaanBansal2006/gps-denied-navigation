# Decision 033: Cross-Sequence Validation of RLS Adaptation

**Date:** 2026-05-24 (scaffold; numbers to be filled in when desktop eval completes)
**Status:** Pending — cross_sequence_eval.py running on desktop RTX 2060.

## Context

Decision 029 reported a **36 % improvement** at the 30-second headline
metric from v15 + filter (0.403 m/s) to v15 + filter + RLS (0.259 m/s),
but flagged a real limitation: the RLS hyperparameters (`p_init=0.1`,
`forgetting=0.995`) were *selected on the same MH_05 test sequence they
were reported on*. There's no guarantee the win replicates elsewhere.

This decision tests the cross-sequence claim by running the same RLS
configuration (no further tuning) on three additional EuRoC sequences:

- **MH_03_medium** — same machine-hall environment, less aggressive motion.
- **V1_03_difficult** — different environment (Vicon room), aggressive motion.
- **MH_04_difficult** — the val sequence (in case it's actually
  representative of test).

Same configurations also tested:
- Vanilla v15 + filter (baseline)
- v15 + filter + Continuous adapter (decision 032)

## Results — TO BE FILLED IN

| Sequence | Vanilla v15 | + RLS | Δ vs vanilla | + Continuous | Δ vs vanilla |
|---|---:|---:|---:|---:|---:|
| MH_03_medium | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| V1_03_difficult | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MH_04_difficult | 0.782 | 0.750 | -4 % | 0.819 | +5 % |
| MH_05_difficult (test) | 0.403 | 0.259 | -36 % | 0.246 | -39 % |

(Final velocity error at 30s outage, m/s.)

## Findings — TO BE FILLED IN

### Does the 36 % win replicate?
_TBD: per-sequence verdict. Strong replication → headline holds. Partial
replication → fold into the "limitations" section. No replication → the
36 % was a single-window artifact._

### Does continuous adaptation replicate its test-set win?
_TBD: on MH_04 it lost; what happens on MH_03 / V1_03? If it consistently
loses on harder/different sequences, the test win is methodologically suspect._

### Per-sequence character
_TBD: which sequences favor which adapter? Motion-class correlation if any._

## Decision — TO BE FILLED IN

_TBD: if RLS holds: keep headline as-is. If RLS doesn't hold: update
README with a "MH_05-only" caveat and demote the headline to "best
configuration tested" rather than "winning method." Decision 032's
val/test conflict warning becomes more acute._

## What this changes about the project

- New eval script: `scripts/cross_sequence_eval.py`.
- New results file: `results/cross_sequence/test_metrics.json`.
- README ablation table may grow a "cross-sequence variance" row.

## What's left

Even with three additional EuRoC sequences, all training data comes from
the same MAV rig. Genuine cross-dataset eval (TUM-VI, KITTI) is the
proper test of generalization — queued post-sprint.
