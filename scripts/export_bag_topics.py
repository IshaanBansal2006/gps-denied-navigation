#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import rosbag


OUTPUT_DIR = Path("data/processed")
IMU_OUTPUT = OUTPUT_DIR / "imu.csv"
LEICA_OUTPUT = OUTPUT_DIR / "leica_position.csv"

DEFAULT_BAGS = [Path("data/MH_01_easy.bag")]


def stamp_to_float_secs(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def export_bags(bag_paths: list[Path]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for bag_path in bag_paths:
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

    with open(IMU_OUTPUT, "w", newline="") as imu_f, \
         open(LEICA_OUTPUT, "w", newline="") as leica_f:

        imu_writer = csv.writer(imu_f)
        leica_writer = csv.writer(leica_f)

        imu_writer.writerow([
            "timestamp", "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z", "frame_id", "seq",
        ])
        leica_writer.writerow([
            "timestamp", "pos_x", "pos_y", "pos_z", "frame_id", "seq",
        ])

        time_offset = 0.0

        for bag_path in bag_paths:
            print(f"Reading: {bag_path}")
            bag_end_time = 0.0

            with rosbag.Bag(str(bag_path), "r") as bag:
                for _, msg, _ in bag.read_messages(topics=["/imu0"]):
                    t = stamp_to_float_secs(msg.header.stamp) + time_offset
                    imu_writer.writerow([
                        t,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                        msg.header.frame_id,
                        msg.header.seq,
                    ])
                    bag_end_time = max(bag_end_time, t)

                for _, msg, _ in bag.read_messages(topics=["/leica/position"]):
                    t = stamp_to_float_secs(msg.header.stamp) + time_offset
                    leica_writer.writerow([
                        t,
                        msg.point.x,
                        msg.point.y,
                        msg.point.z,
                        msg.header.frame_id,
                        msg.header.seq,
                    ])
                    bag_end_time = max(bag_end_time, t)

            time_offset = bag_end_time + 1.0  # 1s gap between bags

    print(f"IMU CSV → {IMU_OUTPUT}")
    print(f"Leica CSV → {LEICA_OUTPUT}")
    print("Done.")


def main() -> None:
    bag_paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_BAGS
    export_bags(bag_paths)


if __name__ == "__main__":
    main()
