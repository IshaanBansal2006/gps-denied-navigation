# Decision 013: Dead Reckoning Baseline + EKF State Vector

## Context
- Ran IMU-only dead reckoning on MH_05_difficult (test sequence)
- Naive integration: subtract estimated accel bias, integrate accel → velocity
- Result: window delta_v MSE = 34.5, velocity error 49 m/s after just 5 seconds

## Dead Reckoning Results (MH_05_difficult)
| Metric | Value |
|---|---|
| Vel error at 5s | 49.2 m/s |
| Vel error at 60s | 606.2 m/s |
| Window delta_v MSE | 34.519 |
| TCN MSE (same test set) | 0.089 |
| TCN improvement | 387x better than dead reckoning |

## Root cause of catastrophic dead reckoning drift
Gravity subtraction requires knowing attitude (roll/pitch) at each timestep to rotate
gravity from world frame to body frame. Without attitude, even a small error in gravity
direction accumulates unboundedly. The naive bias estimate from first-N-samples is also
invalid when the drone is already in motion at sequence start.

## Decision
The EKF must maintain attitude as part of its state. Use a 15-state vector:
- position (3): px, py, pz
- velocity (3): vx, vy, vz
- attitude (3): roll, pitch, yaw (small-angle linearization for EKF)
- accel bias (3): bax, bay, baz
- gyro bias (3): bgx, bgy, bgz

This is the standard strapdown INS state for IMU-based navigation.

## Consequences
- Prior plan used a simplified 6-state EKF (velocity + accel_bias only) — REVISED to 15-state
- `src/filters/ekf.py`: implement full strapdown INS mechanization as process model
- Gravity subtraction: rotate [0, 0, g] from world to body using current attitude estimate, subtract
- TCN output (delta_v) feeds in as velocity measurement update (no attitude correction needed for this)
- The TCN is still 387x better than raw dead reckoning for predicting window delta_v
