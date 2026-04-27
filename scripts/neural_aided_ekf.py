#!/usr/bin/env python3
"""
Neural-aided navigation: compare drift during simulated GPS outages.

Three modes:
  1. Dead reckoning (IMU integration, no filter)
  2. TCN velocity (accumulated TCN delta_v predictions — no strapdown gravity issue)
  3. EKF + GPS (upper bound — Leica velocity always available)

Architecture note: strapdown INS without GPS drifts 30+ m/s within 5 s due to
uncorrected gravity (requires known attitude). TCN mode bypasses this by using
TCN delta_v predictions as the primary velocity estimator during outages.

Usage:
    python3 scripts/neural_aided_ekf.py [--seq SEQ] [--checkpoint PATH] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.filters.ekf import EKF15, init_from_static
from src.models.tcn import TCNRegressor

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "neural_aided_ekf"

WINDOW_SIZE = 200   # samples (1 s at 200 Hz)
STRIDE = 25         # step between consecutive TCN windows (0.125 s)

OUTAGE_DURATIONS_S = [5, 10, 30, 60]


def load_tcn(checkpoint: str, device: torch.device) -> TCNRegressor:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = TCNRegressor(
        input_channels=cfg.get("input_channels", 6),
        output_dim=cfg.get("output_dim", 3),
        channel_sizes=cfg.get("channel_sizes", [32, 64, 64]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_norm_stats() -> tuple[np.ndarray, np.ndarray]:
    with open(SPLITS_DIR / "normalization_stats.json") as f:
        ns = json.load(f)
    mean = np.array(ns["feature_mean"], dtype=np.float32)
    std = np.array(ns["feature_std"], dtype=np.float32)
    return mean, std


def predict_delta_v(
    model: TCNRegressor,
    window: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = (window.astype(np.float32) - mean) / std          # (200, 6)
    t = torch.tensor(x[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy()[0]                   # (3,)


def run_dead_reckoning(
    accel: np.ndarray,
    timestamps: np.ndarray,
    ba: np.ndarray,
    vel_init: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """IMU integration: subtract bias, no gravity removal."""
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    for i in range(n):
        k = start + i
        dt = timestamps[k + 1] - timestamps[k]
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        vel[i + 1] = vel[i] + (accel[k] - ba) * dt
    return vel


def run_tcn_velocity(
    model: TCNRegressor,
    imu_data: np.ndarray,
    timestamps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    vel_init: np.ndarray,
    start: int,
    end: int,
    abs_start: int,   # absolute index in imu_data where sequence begins (for window lookup)
) -> np.ndarray:
    """
    TCN-based velocity estimation during GPS outage.

    At each STRIDE step, run TCN on the last WINDOW_SIZE IMU samples to predict
    delta_v. Velocity is updated as:
        v_now = v_200_samples_ago + delta_v_tcn
    where v_200_samples_ago comes from a rolling velocity history buffer.

    This avoids gravity integration entirely — the TCN handles dynamics implicitly.
    """
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init

    # Rolling buffer: velocity estimate at each absolute sample index
    # Pre-fill the buffer with vel_init for the period before the outage
    v_buffer: deque[np.ndarray] = deque(
        [vel_init.copy() for _ in range(WINDOW_SIZE + 1)],
        maxlen=WINDOW_SIZE + 1,
    )

    v_current = vel_init.copy()
    steps_since_update = 0

    for i in range(n):
        k = start + i  # absolute index
        v_buffer.append(v_current.copy())

        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            # Window: last WINDOW_SIZE samples in imu_data
            win_end = k + 1
            win_start = win_end - WINDOW_SIZE
            if win_start < 0:
                vel[i + 1] = v_current
                continue
            window = imu_data[win_start:win_end]
            delta_v = predict_delta_v(model, window, mean, std, device)

            # v_now = v_200_ago + delta_v (TCN predicted velocity change)
            v_200_ago = v_buffer[0]   # oldest entry = WINDOW_SIZE steps back
            v_current = v_200_ago + delta_v

        vel[i + 1] = v_current

    return vel


def run_ekf_gps(
    accel: np.ndarray,
    gyro: np.ndarray,
    gt_vel: np.ndarray,
    timestamps: np.ndarray,
    accel_static: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """EKF with GPS at every step (upper bound)."""
    ekf = init_from_static(accel_static, gt_vel[start])
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = gt_vel[start]
    for i in range(n):
        k = start + i
        dt = timestamps[k + 1] - timestamps[k]
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[k], gyro[k], dt)
        ekf.update_velocity(gt_vel[k + 1])
        vel[i + 1] = ekf.s.v.copy()
    return vel


def evaluate_outage(
    df: pd.DataFrame,
    model: TCNRegressor,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    outage_s: float,
    n_static: int = 200,
) -> dict:
    timestamps = df["timestamp"].to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()
    imu_data = df[["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]].to_numpy()

    accel_static = accel[:n_static].mean(axis=0)
    duration = timestamps[-1] - timestamps[0]
    outage_start_t = duration * 0.4
    outage_start = int(np.searchsorted(timestamps - timestamps[0], outage_start_t))
    outage_end = min(
        len(df) - 1,
        int(np.searchsorted(timestamps - timestamps[0], outage_start_t + outage_s))
    )

    # Warm-up EKF to get good bias estimate at outage start
    ekf_warm = init_from_static(accel_static, gt_vel[0])
    for i in range(1, outage_start + 1):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf_warm.predict(accel[i], gyro[i], dt)
        ekf_warm.update_velocity(gt_vel[i])

    ba_est = ekf_warm.s.ba.copy()
    vel_at_outage = gt_vel[outage_start]
    gt_outage = gt_vel[outage_start:outage_end + 1]

    # --- Dead reckoning ---
    vel_dr = run_dead_reckoning(accel, timestamps, ba_est, vel_at_outage, outage_start, outage_end)

    # --- TCN velocity ---
    vel_tcn = run_tcn_velocity(
        model, imu_data, timestamps, mean, std, device,
        vel_at_outage, outage_start, outage_end, abs_start=0,
    )

    # --- EKF + GPS upper bound ---
    vel_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, outage_start, outage_end)

    def final_err(v_traj):
        n = min(len(v_traj), len(gt_outage)) - 1
        return float(np.linalg.norm(v_traj[n] - gt_outage[n]))

    def mean_err(v_traj):
        n = min(len(v_traj), len(gt_outage))
        return float(np.mean(np.linalg.norm(v_traj[:n] - gt_outage[:n], axis=1)))

    return {
        "outage_s": outage_s,
        "outage_samples": outage_end - outage_start,
        "final_vel_error_dr": final_err(vel_dr),
        "final_vel_error_tcn": final_err(vel_tcn),
        "final_vel_error_ekf_gps": final_err(vel_gps),
        "mean_vel_error_dr": mean_err(vel_dr),
        "mean_vel_error_tcn": mean_err(vel_tcn),
        "mean_vel_error_ekf_gps": mean_err(vel_gps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_01_easy")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "tcn_multi.pt"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_tcn(args.checkpoint, device)
    mean, std = load_norm_stats()

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"Sequence: {args.seq}  |  {len(df)} samples")

    all_results = []
    print(f"\n{'Outage':>8} {'DR final err':>14} {'TCN final':>12} {'EKF+GPS':>10} {'TCN vs DR':>12}")
    print("-" * 60)

    for t_out in OUTAGE_DURATIONS_S:
        r = evaluate_outage(df, model, mean, std, device, t_out)
        all_results.append(r)
        dr = r["final_vel_error_dr"]
        tcn = r["final_vel_error_tcn"]
        gps = r["final_vel_error_ekf_gps"]
        improvement = (dr - tcn) / dr * 100 if dr > 0 else 0.0
        print(f"{t_out:>6}s  {dr:>12.3f}  {tcn:>10.3f}  {gps:>8.3f}  {improvement:>+10.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.seq}_outage_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"seq": args.seq, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
