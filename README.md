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

Project planning and system architecture design.