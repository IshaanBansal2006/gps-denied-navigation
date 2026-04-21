# Decision 019: Velocity-Only Filter Beats Full Strapdown EKF During GPS Outage

**Date:** 2026-04-21
**Status:** Accepted

## Context

The natural next step after v7 was to feed TCN absolute velocity predictions into the EKF as
velocity measurements during GPS outage. The EKF would propagate with IMU between TCN updates
and smooth them via Kalman gain.

Two architectures were tested:
- **Full strapdown EKF + TCN**: EKF.predict() at 200 Hz, EKF.update_velocity(tcn) every STRIDE=25 steps
- **Velocity-only filter**: constant-velocity random walk (P += sigma_process² * I * dt), no IMU propagation

sigma_tcn sweep for strapdown: [0.01, 0.05, 0.1, 0.3 m/s]

## Results (MH_05_difficult, 30s outage)

| Filter | Final error | Mean error |
|---|---|---|
| Dead reckoning | 45.867 | 23.181 |
| TCN v7 standalone | 0.501 | 0.974 |
| **Velocity-only filter** | **0.440** | **0.962** |
| Strapdown EKF σ=0.01 | 0.714 | 1.385 |
| Strapdown EKF σ=0.05 | 3.096 | 3.585 |
| EKF + GPS (upper bound) | 0.104 | 0.202 |

## Root Cause: Attitude Drift During GPS Outage

The strapdown EKF propagates velocity as `v += (R @ a_corrected + g) * dt`. This requires
a correct rotation matrix R (attitude). During GPS outage, R is integrated from gyro only.
With sigma_bg ≈ 1e-2 rad/s typical for EuRoC, after 30s attitude error ≈ 0.3 rad, causing
~2.9 m/s² of gravity subtraction error. Over 0.125s between TCN updates: ~0.36 m/s of
contamination per update.

The Kalman gain makes this worse: after GPS warm-up, P_v ≈ 4×10⁻⁴. With R_tcn = (0.01)²
= 1×10⁻⁴, gain K ≈ 0.8 — large enough that the TCN pulls the EKF hard, but the IMU
immediately re-introduces drift in the next predict() call. No sigma_tcn value consistently
beats standalone TCN because the IMU propagation is adding more error than TCN can remove.

At short outages (5s), strapdown EKF σ=0.01 slightly beats standalone TCN (0.453 vs 0.476)
because attitude hasn't had time to drift. At 30s+ it falls apart.

## Decision

During GPS outage, use a **velocity-only filter** instead of strapdown propagation:
- State: velocity (3D only)
- Propagation: P += sigma_process² * I * dt (random walk, no IMU)
- Update: standard Kalman with R_tcn = diag([0.61, 0.67, 0.25])² per axis

This gives 12% better final error at 30s vs standalone TCN (0.440 vs 0.501 m/s), and
1.2% better mean error (0.962 vs 0.974 m/s), without any IMU integration.

## Remaining Gap

Best system: velocity-only filter → 0.440 m/s final at 30s.
GPS upper bound: 0.104 m/s.
Gap: ~4x. Closing this requires either more training data or a better TCN (R²=0.095 → higher).

## Architecture Going Forward

- Pre-outage: full strapdown EKF with GPS updates (attitude well-tracked)
- During outage: velocity-only Kalman filter, TCN v7 as measurement source
- GPS restore: reinitialize full EKF from velocity-filter state + static IMU attitude
