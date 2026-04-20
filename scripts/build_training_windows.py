#!/usr/bin/env python3

from pathlib import Path

import csv
import numpy as np
import pandas as pd


INPUT_CSV = Path("data/processed/imu_aligned_with_leica_velocity.csv")

X_OUTPUT = Path("data/processed/X_windows.npy")
Y_OUTPUT = Path("data/processed/y_delta_v.npy")
META_OUTPUT = Path("data/processed/window_metadata.csv")

WINDOW_SIZE = 200   # 1 second at 200 Hz
STRIDE = 25

IMU_FEATURE_COLS = [
    "gyro_x", "gyro_y", "gyro_z",
    "accel_x", "accel_y", "accel_z",
]

GT_VEL_COLS = [
    "gt_vel_x", "gt_vel_y", "gt_vel_z",
]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    n_rows = len(df)
    if n_rows < WINDOW_SIZE:
        raise ValueError(
            f"Not enough rows ({n_rows}) for WINDOW_SIZE={WINDOW_SIZE}"
        )

    X_list = []
    y_list = []
    meta_rows = []

    max_start = n_rows - WINDOW_SIZE

    for start_idx in range(0, max_start + 1, STRIDE):
        end_idx = start_idx + WINDOW_SIZE - 1

        window_df = df.iloc[start_idx:end_idx + 1]

        # Input window: shape (WINDOW_SIZE, 6)
        x_window = window_df[IMU_FEATURE_COLS].to_numpy(dtype=np.float32)

        # Target: delta velocity across the window
        start_vel = df.loc[start_idx, GT_VEL_COLS].to_numpy(dtype=np.float32)
        end_vel = df.loc[end_idx, GT_VEL_COLS].to_numpy(dtype=np.float32)
        delta_v = end_vel - start_vel

        X_list.append(x_window)
        y_list.append(delta_v)

        meta_rows.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": float(df.loc[start_idx, "timestamp"]),
            "end_time": float(df.loc[end_idx, "timestamp"]),
        })

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    X_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_OUTPUT, X)
    np.save(Y_OUTPUT, y)

    with open(META_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["start_idx", "end_idx", "start_time", "end_time"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    print("Window dataset construction complete.")
    print(f"Input CSV rows: {n_rows}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Number of samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved X to: {X_OUTPUT}")
    print(f"Saved y to: {Y_OUTPUT}")
    print(f"Saved metadata to: {META_OUTPUT}")
    print()

    print("Example target rows:")
    print(y[:5])


if __name__ == "__main__":
    main()