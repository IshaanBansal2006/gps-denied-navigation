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