#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


IMU_CSV = Path("data/processed/imu.csv")
LEICA_POS_CSV = Path("data/processed/leica_position.csv")
LEICA_VEL_CSV = Path("data/processed/leica_velocity.csv")


def check_monotonic_timestamps(df: pd.DataFrame, name: str) -> None:
    dt = df["timestamp"].diff().dropna()

    if (dt <= 0).any():
        bad_count = int((dt <= 0).sum())
        raise ValueError(f"{name}: found {bad_count} non-increasing timestamp differences.")

    print(f"{name}: timestamps are strictly increasing.")
    print(f"{name}: rows = {len(df)}")
    print(f"{name}: mean dt = {dt.mean():.6f} s")
    print(f"{name}: min dt = {dt.min():.6f} s")
    print(f"{name}: max dt = {dt.max():.6f} s")
    print(f"{name}: approx frequency = {1.0 / dt.mean():.3f} Hz")
    print()


def derive_velocity(leica_df: pd.DataFrame) -> pd.DataFrame:
    df = leica_df.copy()

    dt = df["timestamp"].diff()
    df["vel_x"] = df["pos_x"].diff() / dt
    df["vel_y"] = df["pos_y"].diff() / dt
    df["vel_z"] = df["pos_z"].diff() / dt

    # first row has no previous sample, so velocity is undefined there
    df = df.dropna(subset=["vel_x", "vel_y", "vel_z"]).reset_index(drop=True)

    return df


def main() -> None:
    if not IMU_CSV.exists():
        raise FileNotFoundError(f"Missing file: {IMU_CSV}")
    if not LEICA_POS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {LEICA_POS_CSV}")

    imu_df = pd.read_csv(IMU_CSV)
    leica_df = pd.read_csv(LEICA_POS_CSV)

    print("=== Timestamp Checks ===")
    check_monotonic_timestamps(imu_df, "IMU")
    check_monotonic_timestamps(leica_df, "Leica Position")

    print("=== Deriving Leica Velocity ===")
    leica_vel_df = derive_velocity(leica_df)

    print(f"Derived Leica velocity rows = {len(leica_vel_df)}")
    print("Velocity column summary:")
    print(leica_vel_df[["vel_x", "vel_y", "vel_z"]].describe())
    print()

    LEICA_VEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    leica_vel_df.to_csv(LEICA_VEL_CSV, index=False)

    print(f"Saved derived velocity CSV to: {LEICA_VEL_CSV}")


if __name__ == "__main__":
    main()