# Decision 007: — First full TCN baseline results

## Context
The first full baseline training run was executed after:
- completing IMU/Leica preprocessing
- generating 200-sample windows
- building chronological train/val/test splits
- validating the pipeline with a successful subset overfit test

The baseline configuration used:
- model: TCN
- input: 200 x 6 IMU window
- target: delta velocity (3)
- epochs: 50
- batch size: 64
- optimizer: Adam
- loss: MSE
- checkpoint policy: best validation checkpoint

## Results
Best validation performance occurred at epoch 1.

- best epoch: 1
- best validation loss: 0.04398194
- test MSE: 0.05011975
- test MAE: 0.16644402

Axis-wise test metrics:
- mse_x: 0.02855728
- mse_y: 0.05025956
- mse_z: 0.07154243
- mae_x: 0.11777277
- mae_y: 0.17699020
- mae_z: 0.20456913

## Interpretation
The full training pipeline is functioning correctly, including:
- train/validation tracking
- best-checkpoint saving
- test-set evaluation
- result artifact generation

However, the model begins overfitting very early. Training loss decreases steadily across epochs, but validation loss does not improve beyond epoch 1.

## Decision
Treat this run as the project’s first formal baseline result.

Do not change the dataset pipeline yet. The next step should focus on improving training behavior and generalization before moving on to EKF integration.

## Consequences
Future work should prioritize:
- plotting train vs validation loss
- inspecting target distributions and prediction scales
- trying training improvements such as early stopping, lower learning rate, stronger regularization, or smaller model capacity
- preserving this run as the baseline for comparison