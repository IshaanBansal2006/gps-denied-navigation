# System Architecture

## Overview

This project implements a neural-aided inertial navigation system for GPS-denied environments.

The system combines:

- IMU measurements
- Neural network motion prediction
- Extended Kalman Filter

## High-Level Architecture

IMU → Window Builder → Neural Network → EKF → Output State

Optional GPS input:

GPS → EKF → Output State

## Components

### IMU

Provides:

- Accelerometer data
- Gyroscope data

Used for motion propagation.

### Window Builder

Converts raw IMU stream into fixed-length windows.

Example:

200 samples per window

### Neural Network

Input:
- IMU window

Output:
- Delta velocity estimate

Purpose:
- Learn motion patterns
- Reduce drift

### EKF

Maintains:

- Position
- Velocity
- Orientation

Uses:

- IMU propagation
- GPS updates
- Neural pseudo-measurements

## GPS Denied Mode

When GPS is unavailable:

IMU → Neural Network → EKF

The neural network provides pseudo-measurements.