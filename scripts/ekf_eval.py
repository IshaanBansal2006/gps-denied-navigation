#!/usr/bin/env python3
"""
EKF baseline evaluation — full GPS mode.

Runs the 15-state EKF on a sequence with Leica/Vicon velocity measurements
at every IMU step (simulated GPS always available). Establishes the EKF upper
bound before introducing GPS outages.

Usage:
    python3 scripts/ekf_eval.py [--seq SEQ] [--n-static N] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.filters.ekf import EKF15, init_from_static

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RESULTS_DIR = PROJECT_ROOT / "results" / "ekf_eval"


def run_ekf_gps(df: pd.DataFrame, n_static: int = 200) -> dict:
    """Run EKF with full GPS (Leica vel at every step)."""
    timestamps = df["timestamp"].to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()

    accel_static = accel[:n_static].mean(axis=0)
    vel_init = gt_vel[0]

    ekf = init_from_static(accel_static, vel_init)

    n = len(df)
    vel_est = np.zeros((n, 3))
    pos_est = np.zeros((n, 3))
    vel_est[0] = vel_init

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[i], gyro[i], dt)
        ekf.update_velocity(gt_vel[i])        # GPS always available
        vel_est[i] = ekf.s.v.copy()
        pos_est[i] = ekf.s.p.copy()

    return {"vel_est": vel_est, "pos_est": pos_est, "gt_vel": gt_vel}


def compute_metrics(vel_est: np.ndarray, gt_vel: np.ndarray) -> dict:
    diff = vel_est - gt_vel
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    zero_mse = float(np.mean(gt_vel ** 2))
    axis_mse = np.mean(diff ** 2, axis=0).tolist()
    t_var = gt_vel.var(axis=0)
    axis_r2 = [
        float(1 - axis_mse[i] / t_var[i]) if t_var[i] > 1e-8 else float("nan")
        for i in range(3)
    ]
    r2_mean = float(np.nanmean(axis_r2))

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        return float(ac @ bc / d) if d > 1e-8 else 0.0

    corr = [pearson(vel_est[:, i], gt_vel[:, i]) for i in range(3)]

    return {
        "mse": mse, "mae": mae, "r2_mean": r2_mean,
        "zero_predictor_mse": zero_mse,
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
    }


def drift_at_seconds(vel_est: np.ndarray, gt_vel: np.ndarray,
                     timestamps: np.ndarray) -> dict:
    vel_err = np.linalg.norm(vel_est - gt_vel, axis=1)
    t0 = timestamps[0]
    out = {}
    for t_sec in [5, 10, 30, 60]:
        idx = np.searchsorted(timestamps - t0, t_sec)
        if idx < len(vel_err):
            out[f"vel_error_at_{t_sec}s"] = float(vel_err[idx])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_01_easy")
    parser.add_argument("--n-static", type=int, default=200)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"Sequence: {args.seq}  |  {len(df)} samples  |  {df.timestamp.iloc[-1]-df.timestamp.iloc[0]:.1f}s")

    res = run_ekf_gps(df, n_static=args.n_static)
    vel_est, gt_vel = res["vel_est"], res["gt_vel"]

    m = compute_metrics(vel_est, gt_vel)
    drift = drift_at_seconds(vel_est, gt_vel, df["timestamp"].to_numpy())

    print(f"\nEKF (full GPS) metrics:")
    print(f"  MSE:     {m['mse']:.5f}")
    print(f"  R²:      {m['r2_mean']:.4f}")
    print(f"  corr x/y/z: {m['corr_x']:.3f} / {m['corr_y']:.3f} / {m['corr_z']:.3f}")
    print(f"\nVelocity drift (EKF vs GT):")
    for k, v in drift.items():
        print(f"  {k}: {v:.4f} m/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"seq": args.seq, **m, **drift}
    out_path = RESULTS_DIR / f"{args.seq}_gps.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        t = df["timestamp"].to_numpy() - df["timestamp"].iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, lbl in enumerate(["vx", "vy", "vz"]):
            axes[i].plot(t, gt_vel[:, i], "k-", label="GT", linewidth=0.5)
            axes[i].plot(t, vel_est[:, i], "r--", label="EKF", linewidth=0.5)
            axes[i].set_ylabel(lbl + " (m/s)")
            axes[i].legend(fontsize=8)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"EKF velocity (full GPS) — {args.seq}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{args.seq}_gps_velocity.png", dpi=150)
        print(f"Plot saved.")


if __name__ == "__main__":
    main()
