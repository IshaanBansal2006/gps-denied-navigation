# Decision 016: TCN V6 Add V1 03 Difficult

## Context
- TCN v3 (best prior run) trained on 5 sequences (5548 windows): MH_01–03, V1_01–02.
- V1_03_difficult was available on disk but not included in the train split.
- Adding it extends vicon-room coverage with a harder trajectory (837 windows, ~105 s).
- No model, loss, or hyperparameter changes — data-only experiment.

## Decision
Include V1_03_difficult in the training split, growing train windows from 5548 to 6385.

## Reason
- More difficult vicon-room data should improve generalization to MH_05_difficult test set.
- Isolated variable: only the data changes, so any metric shift is attributable to this sequence.
- V1_03_difficult is the hardest V1 sequence and adds trajectory diversity without requiring new preprocessing.

## Consequences
- `scripts/build_dataset.py` SPLIT_CONFIG train list updated to include `V1_03_difficult`.
- `scripts/train_tcn_v6.py` added; outputs to `results/tcn_v6/` and `checkpoints/tcn_v6.pt`.
- Results vs v3:

| Metric | v3 | v6 | Delta |
|---|---|---|---|
| best_epoch | 27 | 44 | +17 |
| test MSE | 0.09086 | 0.08979 | -0.00107 |
| r2_mean | +0.003 | +0.013 | +0.010 |
| corr_x | 0.052 | 0.109 | +0.057 |
| corr_y | -0.009 | 0.001 | +0.010 |
| corr_z | 0.234 | 0.226 | -0.008 |
| zero_predictor_mse | — | 0.09027 | — |

- r2_mean doubled and corr_x doubled; best_epoch +17 indicates better generalization.
- r2_y remains negative (-0.016); y-axis is still the weakest axis and needs attention.
