# Neural-Aided GPS-Denied Navigation for UAVs

## Problem

GPS is commonly used for UAV navigation, but GPS signals can become unreliable or unavailable in many environments such as indoors, urban canyons, and contested environments. When GPS becomes unavailable, UAVs rely on inertial measurements, which drift over time due to sensor noise and bias. This drift leads to inaccurate position estimation and degraded autonomy performance.

## Goal

The goal of this project is to develop a neural-aided inertial navigation system that improves UAV navigation during GPS-denied conditions. A neural network will learn short-term motion from IMU data and provide pseudo-measurements to an Extended Kalman Filter to reduce drift.

## System Overview

The system consists of the following components:

- IMU data input
- Window-based preprocessing
- Neural motion prediction model
- Extended Kalman Filter
- GPS outage simulation

Pipeline:

IMU → Window Builder → Neural Network → EKF → State Estimate

During GPS outages, the neural network will provide motion estimates to assist the EKF.

## Why This Matters

GPS-denied navigation is critical for:

- Autonomous drones
- Aerospace systems
- Defense applications
- Indoor robotics
- Urban navigation

This project explores learning-based inertial navigation, combining classical estimation with machine learning.

## Roadmap

Planned steps:

1. Dataset loading and preprocessing
2. IMU window generation
3. Baseline neural model
4. Motion prediction evaluation
5. EKF integration
6. GPS outage simulation
7. Model improvements

## Current Status

Completed:
- exported `/imu0` and `/leica/position` from `MH_01_easy.bag`
- derived Leica velocity from position using finite differences
- aligned Leica-derived velocity to IMU timestamps using interpolation
- built fixed-length IMU training windows
- saved ML-ready arrays:
  - `data/processed/X_windows.npy`
  - `data/processed/y_delta_v.npy`
- chronologically split dataset into train / validation / test
- normalized IMU features using training-set statistics only

Current split outputs:
- `data/processed/splits/X_train.npy`
- `data/processed/splits/y_train.npy`
- `data/processed/splits/X_val.npy`
- `data/processed/splits/y_val.npy`
- `data/processed/splits/X_test.npy`
- `data/processed/splits/y_test.npy`
- `data/processed/splits/normalization_stats.json`

Training definition:
- window length: 1 second
- 200 IMU samples per window
- input shape per sample: `(200, 6)`
- target shape per sample: `(3,)`
- target: delta velocity across the window

Next step:
- build the first baseline TCN
- train on the normalized chronological split
- verify the model can overfit a small subset before full training

# Local Datasets

This folder contains large datasets that are NOT committed to the repository.

Datasets used:
### EuRoC MAV Dataset

Download from:
https://www.kaggle.com/datasets/chunai/euroc-mh-01-easy-ros-bag-dataset?resource=download

Sequences used:

- MH_01_easy
- (more will be added later)

Expected structure:

data/
    euroc/
        MH_01_easy/
            mav0/
                imu0/
                cam0/
                cam1/
                state_groundtruth_estimate0/

Do not commit dataset files.

## Current Training Definition

Window length:
- 1 second
- 200 IMU samples

Model input:
- IMU window of shape `(200, 6)`

Model target:
- delta velocity across the window:
  - Δvx
  - Δvy
  - Δvz