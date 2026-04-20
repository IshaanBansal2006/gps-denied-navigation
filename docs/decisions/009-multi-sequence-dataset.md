# Decision 009: Multi-Sequence Dataset

## Context
- Single-sequence training (MH_01_easy, ~1465 windows) caused overfitting at epoch 4
- EuRoC provides 8 sequences: MH_01–05 and V1_01–03 covering two environments
- Model needs to generalize across flight dynamics and environments before EKF integration

## Decision
Expand training data to multiple EuRoC sequences using a new per-sequence processing pipeline that stores intermediate CSVs and windows under `data/sequences/<name>/`.

## Reason
- Per-sequence intermediate files enable idempotent re-processing (skip if already done)
- Storing windows separately allows flexible split changes without re-running the full pipeline
- Two environments (Machine Hall + Vicon Room) test cross-environment generalization, which is necessary for the eventual hardware deployment goal

## Consequences
- New script: `scripts/process_sequence.py` — processes one bag end-to-end
- New script: `scripts/build_dataset.py` — assembles sequences into train/val/test splits
- Expected: ~5000–7000 train windows (vs. 1465 before) → significantly later overfitting epoch
- Old `data/processed/` pipeline still works for single-bag debugging
