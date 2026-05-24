"""Render an animated GIF of the hero-figure trajectories building up over the 30 s outage.

Mirrors the static hero figure's eval pipeline, then frame-by-frame draws the
trajectories accumulating in real time. Saves an animated GIF suitable for
embedding in the README.

Output:
  docs/figures/trajectory_animation.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.neural_aided_ekf_lstm_v15 import (
    load_lstm_v15,
    warm_lstm,
    run_dead_reckoning,
    run_vel_only_filter,
    run_ekf_gps,
)
from src.filters.ekf import init_from_static

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

C_TRUTH = "#111111"
C_DR    = "#d62728"
C_V15   = "#1f77b4"
C_GPS   = "#7a7a7a"


def integrate_velocity(v: np.ndarray, dt_arr: np.ndarray) -> np.ndarray:
    n = len(v) - 1
    p = np.zeros((n + 1, 3))
    for i in range(n):
        p[i + 1] = p[i] + 0.5 * (v[i] + v[i + 1]) * dt_arr[i]
    return p


def safe_dt_array(timestamps: np.ndarray, start: int, end: int) -> np.ndarray:
    dt = np.diff(timestamps[start:end + 1])
    return np.where((dt > 0) & (dt < 0.05), dt, 0.005)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outage", type=float, default=30.0)
    parser.add_argument("--outage-start-frac", type=float, default=0.6)
    parser.add_argument("--out", default="trajectory_animation")
    parser.add_argument("--frames", type=int, default=120,
                        help="number of animation frames (default 120 → 8 s at 15 fps)")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm = load_lstm_v15(device)
    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")

    timestamps = df["timestamp"].to_numpy()
    accel  = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro   = df[["gyro_x",  "gyro_y",  "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()
    imu_data = df[["gyro_x", "gyro_y", "gyro_z",
                   "accel_x", "accel_y", "accel_z"]].to_numpy()

    n_static = 200
    accel_static = accel[:n_static].mean(axis=0)
    duration = timestamps[-1] - timestamps[0]
    outage_start_t = duration * args.outage_start_frac
    outage_start = int(np.searchsorted(timestamps - timestamps[0], outage_start_t))
    outage_end = min(len(df) - 1,
                     int(np.searchsorted(timestamps - timestamps[0],
                                         outage_start_t + args.outage)))

    ekf_warm = init_from_static(accel_static, gt_vel[0])
    for i in range(1, outage_start + 1):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf_warm.predict(accel[i], gyro[i], dt)
        ekf_warm.update_velocity(gt_vel[i])
    ba_est = ekf_warm.s.ba.copy()

    print("[lstm] warming hidden state...", end=" ", flush=True)
    _, lstm_state0 = warm_lstm(model, imu_data, norm, device, outage_start)
    print("done")

    vel_at_outage = gt_vel[outage_start]
    v_truth = gt_vel[outage_start:outage_end + 1]
    v_dr  = run_dead_reckoning(accel, timestamps, ba_est, vel_at_outage, outage_start, outage_end)
    v_v15 = run_vel_only_filter(model, imu_data, timestamps, norm, device,
                                vel_at_outage, outage_start, outage_end, lstm_state0)
    v_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static,
                        outage_start, outage_end)

    dt_arr = safe_dt_array(timestamps, outage_start, outage_end)
    p_truth = integrate_velocity(v_truth, dt_arr)
    p_dr    = integrate_velocity(v_dr,    dt_arr)
    p_v15   = integrate_velocity(v_v15,   dt_arr)
    p_gps   = integrate_velocity(v_gps,   dt_arr)
    t = timestamps[outage_start:outage_end + 1] - timestamps[outage_start]

    total = len(p_truth)
    frame_idxs = np.linspace(2, total, args.frames, dtype=int)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(10, 8), dpi=110)

    # Lock axes around the full trajectory extent so frames don't jitter.
    all_xy = np.vstack([p_truth[:, :2], p_v15[:, :2], p_gps[:, :2]])
    pad = 1.5
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East displacement (m)")
    ax.set_ylabel("North displacement (m)")

    # Initialize artists.
    (line_truth,) = ax.plot([], [], color=C_TRUTH, lw=2.2, label="Ground truth (Leica)")
    (line_dr,)    = ax.plot([], [], color=C_DR,    lw=1.8, ls="--", label="IMU dead-reckon")
    (line_gps,)   = ax.plot([], [], color=C_GPS,   lw=1.5, ls=":",  label="EKF + GPS oracle")
    (line_v15,)   = ax.plot([], [], color=C_V15,   lw=2.6,
                            label="LSTM v15 + filter (ours)")
    head_truth = ax.scatter([0], [0], color=C_TRUTH, s=40, zorder=6)
    head_v15   = ax.scatter([0], [0], color=C_V15,   s=40, zorder=6)
    head_gps   = ax.scatter([0], [0], color=C_GPS,   s=22, zorder=6)
    ax.scatter([0], [0], color="black", s=70, zorder=4)
    ax.annotate("GPS lost", xy=(0, 0), xytext=(12, 12),
                textcoords="offset points", fontsize=10, color="#555555")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    title = ax.set_title(
        f"Neural-aided IMU navigation through a {args.outage:.0f}-second GPS outage  ·  EuRoC {args.seq}\n"
        "t = 0.00 s",
        pad=12, fontsize=12,
    )

    def update(frame_n):
        i = frame_idxs[frame_n]
        line_truth.set_data(p_truth[:i, 0], p_truth[:i, 1])
        line_dr.set_data(p_dr[:i, 0],       p_dr[:i, 1])
        line_v15.set_data(p_v15[:i, 0],     p_v15[:i, 1])
        line_gps.set_data(p_gps[:i, 0],     p_gps[:i, 1])
        head_truth.set_offsets([[p_truth[i - 1, 0], p_truth[i - 1, 1]]])
        head_v15.set_offsets([[p_v15[i - 1, 0],     p_v15[i - 1, 1]]])
        head_gps.set_offsets([[p_gps[i - 1, 0],     p_gps[i - 1, 1]]])
        elapsed = t[i - 1]
        title.set_text(
            f"Neural-aided IMU navigation through a {args.outage:.0f}-second GPS outage  ·  EuRoC {args.seq}\n"
            f"t = {elapsed:5.2f} s"
        )
        return line_truth, line_dr, line_v15, line_gps, head_truth, head_v15, head_gps, title

    anim = FuncAnimation(fig, update, frames=len(frame_idxs), interval=1000 // args.fps, blit=False)

    out_path = FIG_DIR / f"{args.out}.gif"
    print(f"[animate] rendering {len(frame_idxs)} frames at {args.fps} fps...", end=" ", flush=True)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print("done")
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
