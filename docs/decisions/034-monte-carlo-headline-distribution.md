# Decision 034: Monte Carlo Distribution of the 30-s Headline

**Date:** 2026-05-24 (scaffold; numbers to be filled in when desktop eval completes)
**Status:** Pending — monte_carlo_eval.py queued on desktop RTX 2060
(runs after cross_sequence_eval.py completes).

## Context

The 0.259 m/s headline (decision 029) and its cross-sequence followup
(decision 033) all report a *single* outage event per sequence — start at
40 % through the sequence, 30 s wide. That's one data point per system per
config. A single-point claim can be lucky.

This decision turns the headline into a distribution: run the same three
systems through **20 random outage start positions** on MH_05 (seed=42),
report mean, std, p50, p95, max. If the p95 is close to the mean, the
headline is robust. If the p95 is much wider, it exposes the
single-point as a lucky window — itself important to know.

## Approach

`scripts/monte_carlo_eval.py` — 20 outage starts drawn uniformly from
`[WARMUP_LEN, n_samples - OUTAGE_LEN - 1]`. For each, runs:
- Vanilla v15 + filter
- v15 + filter + RLS
- v15 + filter + continuous

All at 30 s outage duration.

## Results — TO BE FILLED IN

Final velocity error (m/s) at 30 s outage on MH_05, N=20 windows, seed=42:

| System | Mean | Std | p50 | p95 | Max | Headline (frac=0.4) |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla v15 + filter | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.403 |
| v15 + filter + RLS | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.259 |
| v15 + filter + continuous | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 0.246 |

## Findings — TO BE FILLED IN

### Is the headline robust or lucky?
_TBD: ratio of p95 to mean. If <1.3x, robust. If >2x, the headline window
was an outlier and the project should report mean ± std as the headline._

### Does the RLS-vs-continuous ranking flip across windows?
_TBD: continuous beat RLS on the headline window (0.246 vs 0.259) but
lost on val MH_04 — how often does each rank first across the 20 random
windows? If unstable, the test-set winner is hard to defend._

## Decision — TO BE FILLED IN

_TBD: based on the distribution, decide whether to lead the README
headline with the single-point number, mean ± std, or median._

## What this changes about the project

- New eval script: `scripts/monte_carlo_eval.py`.
- New results file: `results/monte_carlo/test_metrics.json`.
- README headline may shift from `0.259 m/s` to e.g. `0.27 ± 0.05 m/s
  (n=20)` if the std justifies it.

## What's left

A 2-D Monte Carlo (sweep outage start × outage duration) would give the
fullest picture but is overkill for the headline metric. Deferred.
