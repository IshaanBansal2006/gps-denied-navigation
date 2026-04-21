#!/usr/bin/env python3
"""
Compare GPS-outage navigation performance across TCN checkpoints.

Runs three TCN strategies head-to-head on MH_05_difficult:
  - tcn_multi.pt   (delta_v, multi-seq baseline — used in decision 015)
  - tcn_v6.pt      (delta_v, best R²=+0.013)
  - tcn_v7.pt      (absolute velocity, best R²=+0.095)

Plus the dead-reckoning baseline and EKF+GPS upper bound.

Usage:
    python3 scripts/ekf_outage_comparison.py [--seq MH_05_difficult] [--outages 5,10,30,60]
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
RESULTS_DIR = PROJECT_ROOT / "results" / "outage_comparison"

WINDOW_SIZE = 200
STRIDE = 25

CHECKPOINTS = {
    "multi_seq": {
        "path": PROJECT_ROOT / "checkpoints" / "tcn_multi.pt",
        "mode": "delta_v",
        "norm_stats": PROJECT_ROOT / "data" / "splits" / "normalization_stats.json",
        "norm_key": "feature",  # feature_mean / feature_std
    },
    "v6": {
        "path": PROJECT_ROOT / "checkpoints" / "tcn_v6.pt",
        "mode": "delta_v",
        "norm_stats": PROJECT_ROOT / "data" / "splits" / "normalization_stats.json",
        "norm_key": "feature",
    },
    "v7": {
        "path": PROJECT_ROOT / "checkpoints" / "tcn_v7.pt",
        "mode": "abs_vel",
        "norm_stats": PROJECT_ROOT / "results" / "tcn_v7" / "normalization_stats.json",
        "norm_key": "v7",  # x_mean/x_std for input, y_mean/y_std for output
    },
}


def load_model(cfg: dict, device: torch.device) -> TCNRegressor:
    ckpt = torch.load(cfg["path"], map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    model = TCNRegressor(
        input_channels=c.get("input_channels", 6),
        output_dim=c.get("output_dim", 3),
        channel_sizes=c.get("channel_sizes", [16, 32, 32]),
        kernel_size=c.get("kernel_size", 3),
        dropout=c.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_norm(cfg: dict) -> dict:
    with open(cfg["norm_stats"]) as f:
        ns = json.load(f)
    if cfg["norm_key"] == "feature":
        x_mean = np.array(ns["feature_mean"], dtype=np.float32)
        x_std = np.array(ns["feature_std"], dtype=np.float32)
        return {"x_mean": x_mean, "x_std": x_std}
    else:  # v7 layout
        x_mean = np.array(ns["x_mean"], dtype=np.float32).reshape(6)
        x_std = np.array(ns["x_std"], dtype=np.float32).reshape(6)
        y_mean = np.array(ns["y_mean"], dtype=np.float32).reshape(3)
        y_std = np.array(ns["y_std"], dtype=np.float32).reshape(3)
        return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def infer(model, window, norm, device) -> np.ndarray:
    x = (window.astype(np.float32) - norm["x_mean"]) / norm["x_std"]
    t = torch.tensor(x[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy()[0]


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


def run_tcn_delta_v(model, imu_data, timestamps, norm, device, vel_init, start, end):
    """Delta-v mode: v_now = v_200_samples_ago + predicted_delta_v."""
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    v_buffer: deque[np.ndarray] = deque(
        [vel_init.copy() for _ in range(WINDOW_SIZE + 1)], maxlen=WINDOW_SIZE + 1
    )
    v_current = vel_init.copy()
    steps_since_update = 0
    for i in range(n):
        k = start + i
        v_buffer.append(v_current.copy())
        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            win_end = k + 1
            win_start = win_end - WINDOW_SIZE
            if win_start < 0:
                vel[i + 1] = v_current
                continue
            window = imu_data[win_start:win_end]
            delta_v = infer(model, window, norm, device)
            v_200_ago = v_buffer[0]
            v_current = v_200_ago + delta_v
        vel[i + 1] = v_current
    return vel


def run_tcn_abs_vel(model, imu_data, timestamps, norm, device, vel_init, start, end):
    """Absolute-velocity mode: v_now = denormalize(tcn_output)."""
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    v_current = vel_init.copy()
    steps_since_update = 0
    y_mean = norm["y_mean"]
    y_std = norm["y_std"]
    for i in range(n):
        k = start + i
        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            win_end = k + 1
            win_start = win_end - WINDOW_SIZE
            if win_start < 0:
                vel[i + 1] = v_current
                continue
            window = imu_data[win_start:win_end]
            v_norm = infer(model, window, norm, device)
            v_current = v_norm * y_std + y_mean  # denormalize → m/s
        vel[i + 1] = v_current
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


def evaluate_outage(df, models_norm, device, outage_s, n_static=200):
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
        int(np.searchsorted(timestamps - timestamps[0], outage_start_t + outage_s)),
    )

    # Warm-up EKF to get bias at outage start
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

    def final_err(v):
        n = min(len(v), len(gt_outage)) - 1
        return float(np.linalg.norm(v[n] - gt_outage[n]))

    def mean_err(v):
        n = min(len(v), len(gt_outage))
        return float(np.mean(np.linalg.norm(v[:n] - gt_outage[:n], axis=1)))

    result = {"outage_s": outage_s, "outage_samples": outage_end - outage_start}

    # Dead reckoning
    vel_dr = run_dead_reckoning(accel, timestamps, ba_est, vel_at_outage, outage_start, outage_end)
    result["dr_final"] = final_err(vel_dr)
    result["dr_mean"] = mean_err(vel_dr)

    # Each TCN checkpoint
    for name, (model, norm, cfg) in models_norm.items():
        if cfg["mode"] == "delta_v":
            vel_tcn = run_tcn_delta_v(model, imu_data, timestamps, norm, device, vel_at_outage, outage_start, outage_end)
        else:
            vel_tcn = run_tcn_abs_vel(model, imu_data, timestamps, norm, device, vel_at_outage, outage_start, outage_end)
        result[f"{name}_final"] = final_err(vel_tcn)
        result[f"{name}_mean"] = mean_err(vel_tcn)

    # EKF + GPS upper bound
    vel_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, outage_start, outage_end)
    result["ekf_gps_final"] = final_err(vel_gps)
    result["ekf_gps_mean"] = mean_err(vel_gps)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60")
    args = parser.parse_args()

    outage_durations = [float(x) for x in args.outages.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all checkpoints
    models_norm = {}
    for name, cfg in CHECKPOINTS.items():
        if not Path(cfg["path"]).exists():
            print(f"[skip] {name}: checkpoint not found at {cfg['path']}")
            continue
        if not Path(cfg["norm_stats"]).exists():
            print(f"[skip] {name}: norm stats not found at {cfg['norm_stats']}")
            continue
        model = load_model(cfg, device)
        norm = load_norm(cfg)
        models_norm[name] = (model, norm, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {name}: {n_params:,} params | mode={cfg['mode']}")

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"\nSequence: {args.seq}  |  {len(df)} samples")

    all_results = []
    names = list(models_norm.keys())

    # Print header
    header = f"{'Outage':>8}  {'DR':>8}"
    for n in names:
        header += f"  {n:>12}"
    header += f"  {'EKF+GPS':>8}"
    print("\n" + header)
    print("-" * (10 + 10 + 14 * len(names) + 12))

    for t_out in outage_durations:
        r = evaluate_outage(df, models_norm, device, t_out)
        all_results.append(r)

        dr = r["dr_final"]
        line = f"{t_out:>6}s  {dr:>8.3f}"
        for n in names:
            val = r.get(f"{n}_final", float("nan"))
            pct = (dr - val) / dr * 100 if dr > 0 else 0.0
            line += f"  {val:>7.3f} ({pct:>+5.1f}%)"
        gps = r["ekf_gps_final"]
        line += f"  {gps:>8.3f}"
        print(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.seq}_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"seq": args.seq, "checkpoints": list(models_norm.keys()), "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table: mean error at 30s outage
    r30 = next((r for r in all_results if r["outage_s"] == 30), None)
    if r30:
        print("\n--- Mean velocity error at 30s outage ---")
        print(f"  Dead reckoning:  {r30['dr_mean']:.3f} m/s")
        for n in names:
            print(f"  {n:12s}:  {r30.get(f'{n}_mean', float('nan')):.3f} m/s")
        print(f"  EKF+GPS:         {r30['ekf_gps_mean']:.3f} m/s  (upper bound)")


if __name__ == "__main__":
    main()
