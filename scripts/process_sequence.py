#!/usr/bin/env python3
"""Process a single EuRoC bag into windows ready for dataset assembly.

Usage (plain Python 3 — skips rosbag export if CSVs already exist):
    python3 scripts/process_sequence.py <bag_path> <seq_name>

Example:
    python3 scripts/process_sequence.py /home/ishaan/machine_hall/machine_hall/MH_02_easy/MH_02_easy.bag MH_02_easy

Output written to data/sequences/<seq_name>/:
    imu.csv, leica_position.csv, leica_velocity.csv, imu_aligned.csv
    X_windows.npy  (N, 200, 6)
    y_delta_v.npy  (N, 3)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 200
STRIDE = 25

IMU_FEATURE_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
GT_VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]


# ---------------------------------------------------------------------------
# Step 1: export (requires rosbag — skipped if CSVs already present)
# ---------------------------------------------------------------------------

def export_bag(bag_path: Path, seq_dir: Path) -> None:
    imu_csv = seq_dir / "imu.csv"
    leica_csv = seq_dir / "leica_position.csv"

    if imu_csv.exists() and leica_csv.exists():
        print(f"[export] CSVs already exist for {seq_dir.name}, skipping.")
        return

    try:
        import rosbag  # only available in ROS Python 3.8 env
    except ImportError:
        raise RuntimeError(
            f"rosbag not available and CSVs missing for {seq_dir.name}. "
            "Run this script inside the ROS Python environment to export the bag first."
        )

    print(f"[export] Reading {bag_path} ...")
    with rosbag.Bag(str(bag_path), "r") as bag, \
         open(imu_csv, "w", newline="") as imu_f, \
         open(leica_csv, "w", newline="") as leica_f:

        imu_writer = csv.writer(imu_f)
        leica_writer = csv.writer(leica_f)

        imu_writer.writerow([
            "timestamp", "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z", "frame_id", "seq",
        ])
        leica_writer.writerow([
            "timestamp", "pos_x", "pos_y", "pos_z", "frame_id", "seq",
        ])

        for _, msg, _ in bag.read_messages(topics=["/imu0"]):
            t = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9
            imu_writer.writerow([
                t,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.header.frame_id, msg.header.seq,
            ])

        for _, msg, _ in bag.read_messages(topics=["/leica/position", "/vicon/firefly_sbx/firefly_sbx"]):
            t = float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9
            if hasattr(msg, "point"):
                x, y, z = msg.point.x, msg.point.y, msg.point.z
            else:
                x, y, z = msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z
            leica_writer.writerow([t, x, y, z, msg.header.frame_id, msg.header.seq])

    print(f"[export] Done → {imu_csv}, {leica_csv}")


# ---------------------------------------------------------------------------
# Step 2: derive velocity via finite differences
# ---------------------------------------------------------------------------

def derive_velocity(seq_dir: Path) -> None:
    out_path = seq_dir / "leica_velocity.csv"
    if out_path.exists():
        print(f"[velocity] Already exists for {seq_dir.name}, skipping.")
        return

    leica_df = pd.read_csv(seq_dir / "leica_position.csv")
    leica_df = leica_df.sort_values("timestamp").reset_index(drop=True)

    dt = leica_df["timestamp"].diff()
    leica_df["vel_x"] = leica_df["pos_x"].diff() / dt
    leica_df["vel_y"] = leica_df["pos_y"].diff() / dt
    leica_df["vel_z"] = leica_df["pos_z"].diff() / dt
    leica_df = leica_df.dropna(subset=["vel_x", "vel_y", "vel_z"]).reset_index(drop=True)

    leica_df.to_csv(out_path, index=False)
    print(f"[velocity] Saved {len(leica_df)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Step 3: align Leica velocity onto IMU timestamps
# ---------------------------------------------------------------------------

def align_to_imu(seq_dir: Path) -> None:
    out_path = seq_dir / "imu_aligned.csv"
    if out_path.exists():
        print(f"[align] Already exists for {seq_dir.name}, skipping.")
        return

    imu_df = pd.read_csv(seq_dir / "imu.csv").sort_values("timestamp").reset_index(drop=True)
    leica_df = pd.read_csv(seq_dir / "leica_velocity.csv").sort_values("timestamp").reset_index(drop=True)

    leica_t = leica_df["timestamp"].to_numpy()
    imu_t = imu_df["timestamp"].to_numpy()

    valid_mask = (imu_t >= leica_t[0]) & (imu_t <= leica_t[-1])
    imu_valid = imu_df.loc[valid_mask].copy().reset_index(drop=True)
    imu_valid_t = imu_valid["timestamp"].to_numpy()

    for col, src in [("gt_vel_x", "vel_x"), ("gt_vel_y", "vel_y"), ("gt_vel_z", "vel_z")]:
        imu_valid[col] = np.interp(imu_valid_t, leica_t, leica_df[src].to_numpy())

    imu_valid.to_csv(out_path, index=False)
    print(f"[align] {len(imu_valid)} aligned rows → {out_path}")


# ---------------------------------------------------------------------------
# Step 4: build sliding windows
# ---------------------------------------------------------------------------

def build_windows(seq_dir: Path) -> None:
    x_out = seq_dir / "X_windows.npy"
    y_out = seq_dir / "y_delta_v.npy"
    if x_out.exists() and y_out.exists():
        print(f"[windows] Already exists for {seq_dir.name}, skipping.")
        return

    df = pd.read_csv(seq_dir / "imu_aligned.csv")
    n_rows = len(df)

    X_list, y_list = [], []
    for start in range(0, n_rows - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE - 1
        x_win = df.iloc[start:end + 1][IMU_FEATURE_COLS].to_numpy(dtype=np.float32)
        start_vel = df.loc[start, GT_VEL_COLS].to_numpy(dtype=np.float32)
        end_vel = df.loc[end, GT_VEL_COLS].to_numpy(dtype=np.float32)
        X_list.append(x_win)
        y_list.append(end_vel - start_vel)

    X = np.stack(X_list)
    y = np.stack(y_list)
    np.save(x_out, X)
    np.save(y_out, y)
    print(f"[windows] {len(X)} windows (stride={STRIDE}) → {x_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_sequence(bag_path: Path, seq_name: str) -> None:
    seq_dir = Path("data/sequences") / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    export_bag(bag_path, seq_dir)
    derive_velocity(seq_dir)
    align_to_imu(seq_dir)
    build_windows(seq_dir)
    print(f"[done] {seq_name}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/process_sequence.py <bag_path> <seq_name>")
        sys.exit(1)
    process_sequence(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
