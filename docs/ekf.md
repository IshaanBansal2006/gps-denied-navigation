# Extended Kalman Filter

## Purpose

Fuse IMU and neural predictions.

## State

State vector:

- Position
- Velocity
- Orientation

## Propagation

Using IMU:

- Accelerometer
- Gyroscope

## Updates

Normal Mode:

GPS updates

GPS Denied Mode:

Neural pseudo-measurements

## Neural Update

Neural network outputs:

Delta velocity

This is used as a measurement update.

## Goal

Reduce drift during GPS outage.