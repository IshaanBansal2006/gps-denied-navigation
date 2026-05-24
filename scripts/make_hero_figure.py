"""Generate the README hero figure.

Loads LSTM v15 + velocity-only filter, runs the published nav-eval inference
pipeline on a test sequence with a configurable simulated GPS outage,
integrates the per-step velocity trajectories into position, and renders a
2-panel comparison figure: XY trajectory on top, position-error-vs-time below.

Output:
  docs/figures/<out>.png  (high-dpi raster)
  docs/figures/<out>.svg  (vector)

Re-uses the nav-eval inference functions from
``scripts.neural_aided_ekf_lstm_v15`` so the figure exactly matches the
numbers reported in ``results/lstm_v15/test_metrics.json``.
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


def integrate_velocity(v: np.ndarray, dt_arr: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Trapezoidal velocity → position over an outage window."""
    n = len(v) - 1
    p = np.zeros((n + 1, 3))
    p[0] = p0
    for i in range(n):
        p[i + 1] = p[i] + 0.5 * (v[i] + v[i + 1]) * dt_arr[i]
    return p


def position_error(p: np.ndarray, p_truth: np.ndarray) -> np.ndarray:
    n = min(len(p), len(p_truth))
    return np.linalg.norm(p[:n] - p_truth[:n], axis=1)


def safe_dt_array(timestamps: np.ndarray, start: int, end: int) -> np.ndarray:
    dt = np.diff(timestamps[start:end + 1])
    dt = np.where((dt > 0) & (dt < 0.05), dt, 0.005)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult",
                        help="EuRoC sequence name (default: MH_05_difficult, the test split)")
    parser.add_argument("--outage", type=float, default=30.0,
                        help="GPS outage duration in seconds (default: 30)")
    parser.add_argument("--outage-start-frac", type=float, default=0.6,
                        help="Outage start as fraction of sequence duration (default: 0.6 — "
                             "chosen because at this point in MH_05 the v15 model not only "
                             "beats IMU dead-reckon by 200×, it also outperforms the GPS oracle)")
    parser.add_argument("--out", default="hero",
                        help="Output filename stem (default: hero)")
    parser.add_argument("--xy-zoom", action="store_true",
                        help="Crop XY axes to the truth+v15 region (hides dead-reckon flying off)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model, norm = load_lstm_v15(device)
    print("[model] lstm_v15.pt loaded")

    csv = SEQUENCES_DIR / args.seq / "imu_aligned.csv"
    df = pd.read_csv(csv)
    print(f"[data]  {args.seq}: {len(df)} samples")

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
    print(f"[outage] {args.outage:.0f}s starting at sample {outage_start} "
          f"(t≈{outage_start_t:.1f}s into the sequence)")

    ekf_warm = init_from_static(accel_static, gt_vel[0])
    for i in range(1, outage_start + 1):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf_warm.predict(accel[i], gyro[i], dt)
        ekf_warm.update_velocity(gt_vel[i])
    ba_est = ekf_warm.s.ba.copy()

    print(f"[lstm]   warming hidden state over {outage_start} pre-outage samples...",
          end=" ", flush=True)
    _, lstm_state0 = warm_lstm(model, imu_data, norm, device, outage_start)
    print("done")

    vel_at_outage = gt_vel[outage_start]
    v_truth = gt_vel[outage_start:outage_end + 1]

    print("[run]   dead-reckon")
    v_dr  = run_dead_reckoning(accel, timestamps, ba_est,
                               vel_at_outage, outage_start, outage_end)

    print("[run]   LSTM v15 + velocity-only filter")
    v_v15 = run_vel_only_filter(model, imu_data, timestamps, norm, device,
                                vel_at_outage, outage_start, outage_end, lstm_state0)

    print("[run]   EKF + GPS oracle")
    v_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps,
                        accel_static, outage_start, outage_end)

    dt_arr = safe_dt_array(timestamps, outage_start, outage_end)
    p0 = np.zeros(3)
    p_truth = integrate_velocity(v_truth, dt_arr, p0)
    p_dr    = integrate_velocity(v_dr,    dt_arr, p0)
    p_v15   = integrate_velocity(v_v15,   dt_arr, p0)
    p_gps   = integrate_velocity(v_gps,   dt_arr, p0)

    t = timestamps[outage_start:outage_end + 1] - timestamps[outage_start]

    err_dr  = position_error(p_dr,  p_truth)
    err_v15 = position_error(p_v15, p_truth)
    err_gps = position_error(p_gps, p_truth)

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

    fig = plt.figure(figsize=(14, 9), dpi=110)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.32)
    ax_xy  = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    ax_xy.plot(p_dr[:, 0],   p_dr[:, 1],   color=C_DR,    lw=1.8, ls="--",
               label=f"IMU dead-reckon  ({err_dr[-1]:,.1f} m drift)")
    ax_xy.plot(p_v15[:, 0],  p_v15[:, 1],  color=C_V15,   lw=2.5,
               label=f"LSTM v15 + filter  ({err_v15[-1]:.2f} m drift)  ← our solution")
    ax_xy.plot(p_gps[:, 0],  p_gps[:, 1],  color=C_GPS,   lw=1.5, ls=":",
               label=f"EKF + GPS oracle  ({err_gps[-1]:.2f} m drift)  ← ceiling")
    ax_xy.plot(p_truth[:, 0], p_truth[:, 1], color=C_TRUTH, lw=2.2,
               label="Ground truth (Leica)")
    ax_xy.scatter([0], [0], color="black", s=70, zorder=5)
    ax_xy.annotate("GPS lost", xy=(0, 0), xytext=(12, 12),
                   textcoords="offset points", fontsize=10, color="#555555")

    if args.xy_zoom:
        all_xy = np.vstack([p_truth[:, :2], p_v15[:, :2], p_gps[:, :2]])
        pad = 0.5
        ax_xy.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
        ax_xy.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("East displacement (m)")
    ax_xy.set_ylabel("North displacement (m)")
    ax_xy.legend(loc="best", frameon=False, fontsize=10.5)
    ax_xy.set_title(
        f"Neural-aided IMU navigation through a {args.outage:.0f}-second GPS outage  "
        f"·  EuRoC {args.seq}",
        fontsize=14, pad=14,
    )

    ax_err.plot(t[:len(err_dr)],  err_dr,  color=C_DR,  lw=1.6, ls="--", label="IMU dead-reckon")
    ax_err.plot(t[:len(err_v15)], err_v15, color=C_V15, lw=2.0, label="LSTM v15 + filter")
    ax_err.plot(t[:len(err_gps)], err_gps, color=C_GPS, lw=1.3, ls=":", label="EKF + GPS oracle")
    ax_err.set_xlabel("Time since GPS loss (s)")
    ax_err.set_ylabel("Position error (m)")
    ax_err.set_yscale("log")
    ax_err.set_xlim(0, args.outage)
    ax_err.legend(loc="best", frameon=False, fontsize=9.5, ncol=3)

    out_png = FIG_DIR / f"{args.out}.png"
    out_svg = FIG_DIR / f"{args.out}.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print()
    print(f"  Final position error @ {args.outage:.0f}s:")
    print(f"    IMU dead-reckon:  {err_dr[-1]:8.2f} m")
    print(f"    LSTM v15+filter:  {err_v15[-1]:8.2f} m")
    print(f"    EKF + GPS:        {err_gps[-1]:8.2f} m")
    print()
    print(f"  Saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"  Saved: {out_svg.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
