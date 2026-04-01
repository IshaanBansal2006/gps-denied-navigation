# Dataset

## Dataset Used

EuRoC MAV Dataset

## Description

The EuRoC MAV dataset contains:

- IMU measurements
- Camera data
- Ground truth pose

For this project, only IMU and ground truth data will be used.

## Sensors

IMU:

- Accelerometer
- Gyroscope

Ground Truth:

- Position
- Velocity
- Orientation

## Frequency

IMU frequency:

~200 Hz

## Input Format

Each IMU measurement:

- ax
- ay
- az
- gx
- gy
- gz

## Labels

The neural network will predict:

Delta velocity over window:

Δvx  
Δvy  
Δvz

## Purpose

The dataset will be used to train the neural motion prediction model.

## Current Dataset Source

Current source format: ROS bag

File:
- data/MH_01_easy.bag

Available topics:
- /imu0
- /leica/position
- /cam0/image_raw
- /cam1/image_raw

Current engineering plan:
1. Export /imu0 to CSV
2. Export /leica/position to CSV
3. Derive velocity from position
4. Build IMU-window to delta-velocity training samples

## Exported CSV Files

The ROS bag is not used directly for model training.

Current preprocessing step exports the following CSV files from `data/MH_01_easy.bag`:

- `data/processed/imu.csv`
- `data/processed/leica_position.csv`

### IMU CSV Columns
- timestamp
- gyro_x
- gyro_y
- gyro_z
- accel_x
- accel_y
- accel_z
- frame_id
- seq

### Leica Position CSV Columns
- timestamp
- pos_x
- pos_y
- pos_z
- frame_id
- seq

## Initial Export Results

Exported topics from `data/MH_01_easy.bag`:

- `/imu0` -> `data/processed/imu.csv`
- `/leica/position` -> `data/processed/leica_position.csv`

### Observed Sampling Behavior

IMU timestamps are spaced at approximately 0.005 s, indicating about 200 Hz.

Leica position timestamps are spaced at approximately 0.046 to 0.05 s, indicating about 20 to 22 Hz.

### Consequence

Because direct velocity ground truth is not available in the ROS bag, velocity will be derived from Leica position using timestamped finite differences.

## IMU / Ground-Truth Alignment

Leica-derived velocity is aligned to the IMU timeline using linear interpolation.

Input files:
- `data/processed/imu.csv`
- `data/processed/leica_velocity.csv`

Output file:
- `data/processed/imu_aligned_with_leica_velocity.csv`

Result:
Each retained IMU row now includes:
- `gt_vel_x`
- `gt_vel_y`
- `gt_vel_z`

Only IMU timestamps inside the Leica velocity time range are kept.