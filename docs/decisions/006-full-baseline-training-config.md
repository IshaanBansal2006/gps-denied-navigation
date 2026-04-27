# Decision 006: — Full baseline TCN training configuration

## Context
The GPS-denied navigation project has completed:
- IMU/Leica preprocessing
- Leica-derived velocity interpolation onto IMU timestamps
- fixed-window dataset generation
- chronological train/val/test split
- train-only normalization
- TCN subset overfit validation

The subset overfit test succeeded, indicating that the data pipeline, target construction, normalization, model architecture, and training loop are functioning correctly.

The next step is to run the first full baseline training experiment on the chronological train/validation split.

## Decision
Use the following configuration for the first full baseline training run:
- Model: TCN baseline
- Input window: 200 IMU samples
- Input features: 6
- Output target: delta velocity (3)
- Epochs: 50
- Batch size: 64
- Optimizer: Adam
- Loss: MSE
- Checkpoint policy: save best validation model
- Evaluation policy: test set used only after training is complete

## Reason
This configuration is intended to produce the first meaningful generalization result for the project while staying computationally manageable.

- 50 epochs is long enough to observe convergence and generalization trends
- batch size 64 should improve throughput on CUDA while remaining stable
- best-validation checkpointing avoids selecting an overfit final epoch
- delayed test-set evaluation preserves an unbiased estimate of performance

## Consequence
This run becomes the project’s first formal baseline experiment.
Its outputs should be used to:
- establish baseline MSE/MAE
- inspect overfitting behavior
- determine whether architecture changes are needed
- guide later EKF pseudo-measurement integration