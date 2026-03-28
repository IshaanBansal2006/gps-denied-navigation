# Model

## Goal

Predict short-term motion from IMU data.

## Input

IMU window:

Shape:

Window Length × 6

Example:

200 × 6

Channels:

- ax
- ay
- az
- gx
- gy
- gz

## Output

Delta velocity:

Δvx  
Δvy  
Δvz  

## Model Type

Temporal Convolutional Network (TCN)

Future models:

- GRU
- LSTM
- Hybrid models

## Loss Function

Mean Squared Error (MSE)

## Training

Dataset:

EuRoC MAV

Split:

- Training
- Validation
- Testing