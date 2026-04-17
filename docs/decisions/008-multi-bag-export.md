# Decision 008: Multi-Bag Export

## Context
- `export_bag_topics.py` was hardcoded to a single bag (`MH_01_easy.bag`)
- MH_01_easy is only ~3 min of data, yielding 1465 windows at stride=25 — too few to prevent overfitting
- Adding more EuRoC sequences requires concatenating multiple bags into single CSVs

## Decision
Refactor `export_bag_topics.py` to accept a list of bag files and concatenate their IMU and Leica outputs, with a 1-second timestamp offset between bags to prevent collisions.

## Reason
- Timestamp offsetting preserves chronological ordering across bags without merging timelines
- CLI args (`sys.argv`) allow one-off runs with any bag combination without touching code
- Default falls back to `MH_01_easy.bag` so existing single-bag usage is unchanged

## Consequences
- `scripts/export_bag_topics.py` now accepts optional positional args: `python3 scripts/export_bag_topics.py data/MH_01_easy.bag data/MH_02_easy.bag ...`
- Re-run full pipeline from step 1 after adding new bags
- Expected: more windows → later overfitting epoch → lower test MSE
