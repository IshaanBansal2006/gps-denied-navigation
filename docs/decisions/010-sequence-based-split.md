# Decision 010: Sequence-Based Train/Val/Test Split

## Context
- Previous split was chronological within a single sequence — train/val windows were temporally adjacent, causing data leakage
- With 8 sequences available, splitting by sequence eliminates leakage entirely
- Need to test both within-environment generalization (harder MH dynamics) and cross-environment generalization (Vicon Room)

## Decision
Assign whole sequences to each split:
- Train: MH_01_easy, MH_02_easy, MH_03_medium, V1_01_easy
- Val: MH_04_difficult (harder MH dynamics — tests within-environment generalization)
- Test: V1_02_medium (Vicon Room — tests cross-environment generalization)

## Reason
- MH_04 as val: same Leica ground-truth system, harder flight → measures if model handles varied dynamics
- V1_02 as test: completely different environment, different ground-truth system (Vicon) → true generalization metric
- MH_05 and V1_03 (both difficult) held out for stress testing after EKF integration

## Consequences
- Split config stored in `data/splits/normalization_stats.json` for full reproducibility
- Normalization stats computed from train sequences only (MH_01–03 + V1_01)
- Test metric (V1_02 MSE) is a much stronger claim than prior within-sequence test split
