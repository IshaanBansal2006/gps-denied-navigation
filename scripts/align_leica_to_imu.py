#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


IMU_CSV = Path("data/processed/imu.csv")
LEICA_VEL_CSV = Path("data/processed/leica_velocity.csv")
OUTPUT_CSV = Path("data/processed/imu_aligned_with_leica_velocity.csv")


def main() -> None:
    if not IMU_CSV.exists():
        raise FileNotFoundError(f"Missing file: {IMU_CSV}")
    if not LEICA_VEL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {LEICA_VEL_CSV}")

    imu_df = pd.read_csv(IMU_CSV)
    leica_df = pd.read_csv(LEICA_VEL_CSV)

    # Sort by timestamp just to be safe
    imu_df = imu_df.sort_values("timestamp").reset_index(drop=True)
    leica_df = leica_df.sort_values("timestamp").reset_index(drop=True)

    imu_t = imu_df["timestamp"].to_numpy()
    leica_t = leica_df["timestamp"].to_numpy()

    leica_start = leica_t[0]
    leica_end = leica_t[-1]

    # Keep only IMU samples inside the Leica velocity time range
    valid_mask = (imu_t >= leica_start) & (imu_t <= leica_end)
    imu_valid = imu_df.loc[valid_mask].copy().reset_index(drop=True)

    imu_valid_t = imu_valid["timestamp"].to_numpy()

    # Interpolate each velocity component onto IMU timestamps
    imu_valid["gt_vel_x"] = np.interp(
        imu_valid_t,
        leica_t,
        leica_df["vel_x"].to_numpy(),
    )
    imu_valid["gt_vel_y"] = np.interp(
        imu_valid_t,
        leica_t,
        leica_df["vel_y"].to_numpy(),
    )
    imu_valid["gt_vel_z"] = np.interp(
        imu_valid_t,
        leica_t,
        leica_df["vel_z"].to_numpy(),
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    imu_valid.to_csv(OUTPUT_CSV, index=False)

    print("Alignment complete.")
    print(f"Original IMU rows: {len(imu_df)}")
    print(f"Leica velocity rows: {len(leica_df)}")
    print(f"Aligned IMU rows: {len(imu_valid)}")
    print(f"Saved to: {OUTPUT_CSV}")
    print()
    print("Aligned timestamp range:")
    print(f"  start = {imu_valid['timestamp'].iloc[0]:.6f}")
    print(f"  end   = {imu_valid['timestamp'].iloc[-1]:.6f}")
    print()
    print("Preview:")
    print(
        imu_valid[
            [
                "timestamp",
                "gyro_x", "gyro_y", "gyro_z",
                "accel_x", "accel_y", "accel_z",
                "gt_vel_x", "gt_vel_y", "gt_vel_z",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()