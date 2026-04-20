#!/usr/bin/env python3
"""IMU-only dead reckoning baseline on the test sequence.

Integrates (accel - bias_estimate) to get velocity, compares against Leica/Vicon
ground truth. Establishes the drift rate the EKF+TCN must beat.

Usage:
    python3 scripts/dead_reckoning.py [--seq SEQ_NAME]

Default sequence: MH_05_difficult (current test set).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCES_DIR = Path("data/sequences")
RESULTS_DIR = Path("results/dead_reckoning")

GRAVITY = 9.81  # m/s²


def load_sequence(seq_name: str) -> pd.DataFrame:
    path = SEQUENCES_DIR / seq_name / "imu_aligned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path} — run process_sequence.py first")
    return pd.read_csv(path)


def estimate_accel_bias(df: pd.DataFrame, n_static: int = 200) -> np.ndarray:
    """Estimate accel bias from the first n_static samples (assumed near-static)."""
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()[:n_static]
    bias = accel.mean(axis=0)
    # Remove gravity (assume z is up — EuRoC convention)
    bias[2] -= GRAVITY
    return bias


def integrate_velocity(df: pd.DataFrame, accel_bias: np.ndarray) -> np.ndarray:
    """Integrate corrected acceleration to get velocity estimate."""
    timestamps = df["timestamp"].to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()

    n = len(df)
    vel_est = np.zeros((n, 3), dtype=np.float64)

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0 or dt > 0.1:
            vel_est[i] = vel_est[i - 1]
            continue
        accel_corr = accel[i] - accel_bias
        vel_est[i] = vel_est[i - 1] + accel_corr * dt

    return vel_est


def compute_window_delta_v(
    vel: np.ndarray, window_size: int = 200, stride: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delta_v per window matching TCN output format."""
    n = len(vel)
    dr_dv, gt_dv = [], []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size - 1
        dr_dv.append(vel[end] - vel[start])

    return np.array(dr_dv, dtype=np.float32)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    zero_mse = float(np.mean(gt ** 2))

    axis_mse = np.mean(diff ** 2, axis=0).tolist()
    axis_mae = np.mean(np.abs(diff), axis=0).tolist()

    gt_var = gt.var(axis=0)
    axis_r2 = [
        float(1.0 - axis_mse[i] / gt_var[i]) if gt_var[i] > 1e-8 else float("nan")
        for i in range(3)
    ]
    r2_mean = float(np.nanmean(axis_r2))

    return {
        "mse": mse, "mae": mae, "r2_mean": r2_mean,
        "zero_predictor_mse": zero_mse,
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
        "mae_x": axis_mae[0], "mae_y": axis_mae[1], "mae_z": axis_mae[2],
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_05_difficult", help="Sequence name under data/sequences/")
    args = parser.parse_args()

    df = load_sequence(args.seq)
    n_samples = len(df)
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"Sequence: {args.seq}  |  {n_samples} samples  |  {duration:.1f}s")

    accel_bias = estimate_accel_bias(df)
    print(f"Estimated accel bias (m/s²): {accel_bias}")

    vel_est = integrate_velocity(df, accel_bias)
    vel_gt = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()

    # --- Velocity drift over time ---
    vel_error = np.abs(vel_est - vel_gt)
    drift_at = {}
    for t_sec in [5, 10, 30, 60]:
        idx = min(int(t_sec * 200), n_samples - 1)  # 200 Hz
        drift_at[f"vel_error_at_{t_sec}s"] = float(np.linalg.norm(vel_error[idx]))

    print("\nVelocity drift (|estimated - ground_truth| norm):")
    for k, v in drift_at.items():
        print(f"  {k}: {v:.4f} m/s")

    # --- Window-level delta_v metrics ---
    dr_dv = compute_window_delta_v(vel_est)
    gt_dv = compute_window_delta_v(vel_gt)

    m = metrics(dr_dv, gt_dv)
    print(f"\nWindow delta_v metrics ({len(dr_dv)} windows, window=200, stride=25):")
    print(f"  MSE:              {m['mse']:.5f}")
    print(f"  MAE:              {m['mae']:.5f}")
    print(f"  R²  (mean):       {m['r2_mean']:.4f}")
    print(f"  Zero-pred MSE:    {m['zero_predictor_mse']:.5f}")
    print(f"  R² x/y/z:         {m['r2_x']:.3f} / {m['r2_y']:.3f} / {m['r2_z']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "seq": args.seq, "n_samples": n_samples, "duration_s": float(duration),
        **drift_at, **{f"window_{k}": v for k, v in m.items()},
    }
    out_path = RESULTS_DIR / f"{args.seq}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
