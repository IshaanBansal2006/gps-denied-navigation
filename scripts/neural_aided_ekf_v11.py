#!/usr/bin/env python3
"""
Neural-aided EKF using TCN v11 absolute velocity predictions.

TCN v11: 2s window (400 samples), 6-layer [16,32,32,32,32,32], RF=253.

Two fusion strategies tested:
  A. Full strapdown EKF + TCN velocity updates (sweeps sigma_tcn)
  B. Velocity-only filter: constant-velocity random walk + TCN updates (no IMU propagation)
     — sidesteps attitude-drift issue entirely

Compares vs dead reckoning, standalone TCN v11, EKF+GPS upper bound, and v7 results.

Usage:
    python3 scripts/neural_aided_ekf_v11.py [--seq SEQ] [--outages 5,10,30,60]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.filters.ekf import EKF15, init_from_static
from src.models.tcn import TCNRegressor

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RESULTS_DIR = PROJECT_ROOT / "results" / "neural_aided_ekf_v11"

WINDOW_SIZE = 400
STRIDE = 25

# sigma_tcn values to sweep for the full strapdown EKF
SIGMA_TCN_SWEEP = [0.01, 0.05, 0.1, 0.3]

# Process noise for velocity-only filter (sigma_v_process in m/s/√s)
SIGMA_PROCESS = 0.5


def load_v11(device: torch.device) -> tuple[TCNRegressor, dict]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / "tcn_v11.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    model = TCNRegressor(
        input_channels=c.get("input_channels", 6),
        output_dim=c.get("output_dim", 3),
        channel_sizes=c.get("channel_sizes", [16, 32, 32, 32, 32, 32]),
        kernel_size=c.get("kernel_size", 3),
        dropout=c.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(PROJECT_ROOT / "results" / "tcn_v11" / "normalization_stats.json") as f:
        ns = json.load(f)
    norm = {
        "x_mean": np.array(ns["x_mean"], dtype=np.float32).reshape(6),
        "x_std": np.array(ns["x_std"], dtype=np.float32).reshape(6),
        "y_mean": np.array(ns["y_mean"], dtype=np.float32).reshape(3),
        "y_std": np.array(ns["y_std"], dtype=np.float32).reshape(3),
    }
    return model, norm


def tcn_vel(model, imu_data, k, norm, device) -> np.ndarray | None:
    win_start = k + 1 - WINDOW_SIZE
    if win_start < 0:
        return None
    x = (imu_data[win_start:k + 1].astype(np.float32) - norm["x_mean"]) / norm["x_std"]
    t = torch.tensor(x[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_norm = model(t).cpu().numpy()[0]
    return y_norm * norm["y_std"] + norm["y_mean"]


def run_dead_reckoning(accel, timestamps, ba, vel_init, start, end):
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        vel[i + 1] = vel[i] + (accel[k] - ba) * dt
    return vel


def run_standalone_tcn(model, imu_data, norm, device, vel_init, start, end):
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    v_current = vel_init.copy()
    steps_since_update = 0
    for i in range(n):
        k = start + i
        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            v = tcn_vel(model, imu_data, k, norm, device)
            if v is not None:
                v_current = v
        vel[i + 1] = v_current
    return vel


def run_strapdown_ekf_tcn(model, imu_data, accel, gyro, timestamps, norm, device,
                           ekf_warm: EKF15, start, end, sigma_tcn: float):
    """Full strapdown EKF with TCN velocity updates. Sweeps sigma_tcn."""
    ekf = copy.deepcopy(ekf_warm)
    R_tcn = np.eye(3) * sigma_tcn ** 2
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = ekf.s.v.copy()
    steps = 0
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[k], gyro[k], dt)
        steps += 1
        if steps >= STRIDE:
            steps = 0
            v = tcn_vel(model, imu_data, k, norm, device)
            if v is not None:
                ekf.update_velocity(v, R_tcn)
        vel[i + 1] = ekf.s.v.copy()
    return vel


def run_vel_only_filter(model, imu_data, timestamps, norm, device, vel_init, start, end):
    """
    Constant-velocity random walk filter + TCN updates.
    No IMU propagation — sidesteps attitude drift entirely.
    State: v (3D). Propagation: P += sigma_process² * I * dt.
    Update: standard velocity update when TCN fires.
    """
    v = vel_init.copy().astype(np.float64)
    # Use per-axis TCN RMSE as measurement noise
    R_tcn = np.diag([0.61**2, 0.67**2, 0.25**2])
    P = np.eye(3) * 0.5  # generous initial uncertainty
    Q_rate = SIGMA_PROCESS ** 2 * np.eye(3)  # process noise per second
    H = np.eye(3)

    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = v
    steps = 0
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005

        # Propagate: constant velocity, growing uncertainty
        P = P + Q_rate * dt

        steps += 1
        if steps >= STRIDE:
            steps = 0
            v_tcn = tcn_vel(model, imu_data, k, norm, device)
            if v_tcn is not None:
                S = P + R_tcn
                K = P @ np.linalg.inv(S)
                v = v + K @ (v_tcn - v)
                P = (np.eye(3) - K) @ P

        vel[i + 1] = v
    return vel


def run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, start, end):
    ekf = init_from_static(accel_static, gt_vel[start])
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = gt_vel[start]
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[k], gyro[k], dt)
        ekf.update_velocity(gt_vel[k + 1])
        vel[i + 1] = ekf.s.v.copy()
    return vel


def evaluate_outage(df, model, norm, device, outage_s, n_static=200):
    timestamps = df["timestamp"].to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()
    imu_data = df[["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]].to_numpy()

    accel_static = accel[:n_static].mean(axis=0)
    duration = timestamps[-1] - timestamps[0]
    outage_start_t = duration * 0.4
    outage_start = int(np.searchsorted(timestamps - timestamps[0], outage_start_t))
    outage_end = min(len(df) - 1,
                     int(np.searchsorted(timestamps - timestamps[0], outage_start_t + outage_s)))

    ekf_warm = init_from_static(accel_static, gt_vel[0])
    for i in range(1, outage_start + 1):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf_warm.predict(accel[i], gyro[i], dt)
        ekf_warm.update_velocity(gt_vel[i])

    ba_est = ekf_warm.s.ba.copy()
    vel_at_outage = gt_vel[outage_start]
    gt_outage = gt_vel[outage_start:outage_end + 1]

    def ferr(v):
        n = min(len(v), len(gt_outage)) - 1
        return float(np.linalg.norm(v[n] - gt_outage[n]))

    def merr(v):
        n = min(len(v), len(gt_outage))
        return float(np.mean(np.linalg.norm(v[:n] - gt_outage[:n], axis=1)))

    r = {"outage_s": outage_s}
    v_dr = run_dead_reckoning(accel, timestamps, ba_est, vel_at_outage, outage_start, outage_end)
    r["dr_final"], r["dr_mean"] = ferr(v_dr), merr(v_dr)

    v_tcn = run_standalone_tcn(model, imu_data, norm, device, vel_at_outage, outage_start, outage_end)
    r["tcn_final"], r["tcn_mean"] = ferr(v_tcn), merr(v_tcn)

    for sig in SIGMA_TCN_SWEEP:
        v = run_strapdown_ekf_tcn(model, imu_data, accel, gyro, timestamps, norm, device,
                                   ekf_warm, outage_start, outage_end, sig)
        key = f"nekf_{sig:.2f}"
        r[f"{key}_final"], r[f"{key}_mean"] = ferr(v), merr(v)

    v_vof = run_vel_only_filter(model, imu_data, timestamps, norm, device,
                                 vel_at_outage, outage_start, outage_end)
    r["vof_final"], r["vof_mean"] = ferr(v_vof), merr(v_vof)

    v_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, outage_start, outage_end)
    r["gps_final"], r["gps_mean"] = ferr(v_gps), merr(v_gps)

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60")
    args = parser.parse_args()

    outage_durations = [float(x) for x in args.outages.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, norm = load_v11(device)
    print(f"Loaded tcn_v11.pt: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"sigma_tcn sweep: {SIGMA_TCN_SWEEP}")
    print(f"Velocity-only filter: sigma_process={SIGMA_PROCESS} m/s/√s\n")

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"Sequence: {args.seq}  |  {len(df)} samples\n")

    all_results = []
    for t_out in outage_durations:
        r = evaluate_outage(df, model, norm, device, t_out)
        all_results.append(r)

    # Print summary table (final error at outage end)
    cols = (["dr", "tcn"]
            + [f"nekf_{s:.2f}" for s in SIGMA_TCN_SWEEP]
            + ["vof", "gps"])
    labels = (["DR", "TCN-v7"]
              + [f"EKF σ={s}" for s in SIGMA_TCN_SWEEP]
              + ["VelFilter", "EKF+GPS"])

    header = f"{'Outage':>8}" + "".join(f"  {l:>12}" for l in labels)
    print(header)
    print("-" * (10 + 14 * len(cols)))
    for r in all_results:
        row = f"{r['outage_s']:>6}s"
        for c in cols:
            row += f"  {r.get(c+'_final', float('nan')):>10.3f}"
        print(row)

    print("\n--- Mean velocity error at 30s outage ---")
    r30 = next((r for r in all_results if r["outage_s"] == 30), None)
    if r30:
        for c, l in zip(cols, labels):
            v = r30.get(c + "_mean", float("nan"))
            print(f"  {l:>14}: {v:.3f} m/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.seq}_results.json"
    with open(out_path, "w") as f:
        json.dump({"seq": args.seq, "sigma_sweep": SIGMA_TCN_SWEEP,
                   "sigma_process_vof": SIGMA_PROCESS, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
