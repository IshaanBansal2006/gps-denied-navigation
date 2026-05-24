"""EuRoC MAV dataset loader.

One-class abstraction over the per-sequence ``imu_aligned.csv`` files produced
by the project's preprocessing pipeline (``scripts/export_bag_topics.py`` →
``scripts/derive_leica_velocity.py`` → ``scripts/align_leica_to_imu.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


IMU_COLS = ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z")
VEL_COLS = ("gt_vel_x", "gt_vel_y", "gt_vel_z")

EUROC_TRAIN_SEQS = ("MH_01_easy", "MH_02_easy", "MH_03_medium",
                    "V1_01_easy", "V1_02_medium", "V1_03_difficult")
EUROC_VAL_SEQ = "MH_04_difficult"
EUROC_TEST_SEQ = "MH_05_difficult"


@dataclass
class EuRoCSequence:
    """One EuRoC sequence with IMU + ground-truth velocity at 200 Hz.

    Attributes
    ----------
    name: str
        Sequence name, e.g. ``"MH_05_difficult"``.
    timestamps: np.ndarray, shape (N,)
        Seconds-since-epoch IMU timestamps.
    imu: np.ndarray, shape (N, 6)
        Body-frame gyro_xyz + accel_xyz, raw units (rad/s, m/s²).
    gt_vel: np.ndarray, shape (N, 3)
        World-frame ground-truth velocity from Leica/Vicon, m/s.
    """

    name: str
    timestamps: np.ndarray
    imu: np.ndarray
    gt_vel: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @classmethod
    def load(cls, name: str, sequences_dir: Path | str) -> "EuRoCSequence":
        """Load a single sequence from disk."""
        sequences_dir = Path(sequences_dir)
        csv = sequences_dir / name / "imu_aligned.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Sequence not found: {csv}")
        df = pd.read_csv(csv)
        missing = [c for c in IMU_COLS + VEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        return cls(
            name=name,
            timestamps=df["timestamp"].to_numpy(dtype=np.float64),
            imu=df[list(IMU_COLS)].to_numpy(dtype=np.float32),
            gt_vel=df[list(VEL_COLS)].to_numpy(dtype=np.float32),
        )

    @classmethod
    def load_many(
        cls,
        names: Iterable[str],
        sequences_dir: Path | str,
        skip_missing: bool = True,
    ) -> List["EuRoCSequence"]:
        """Load multiple sequences. Missing ones are skipped with a warning."""
        out: List["EuRoCSequence"] = []
        for n in names:
            try:
                out.append(cls.load(n, sequences_dir))
            except FileNotFoundError:
                if not skip_missing:
                    raise
                print(f"[skip] {n}: not found")
        return out

    def outage_window(self, start_frac: float, duration_s: float) -> tuple:
        """Return (outage_start_idx, outage_end_idx) for a fractional start."""
        t0 = self.timestamps[0]
        dt = self.timestamps - t0
        start_t = self.duration_s * start_frac
        start_idx = int(np.searchsorted(dt, start_t))
        end_idx = min(self.n_samples - 1, int(np.searchsorted(dt, start_t + duration_s)))
        return start_idx, end_idx


def normalize(
    imu: np.ndarray,
    gt_vel: np.ndarray,
    norm: dict,
) -> tuple:
    """Apply train-set normalization stats to (IMU, velocity) arrays."""
    imu_n = (imu - norm["x_mean"]) / norm["x_std"]
    vel_n = (gt_vel - norm["y_mean"]) / norm["y_std"]
    return imu_n.astype(np.float32), vel_n.astype(np.float32)
