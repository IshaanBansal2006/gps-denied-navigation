"""Unit tests for gps_denied_nav.data.EuRoCSequence."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gps_denied_nav.data import IMU_COLS, VEL_COLS, EuRoCSequence


@pytest.fixture
def synthetic_sequence_dir():
    """Build a tiny synthetic sequence on disk with the imu_aligned.csv schema."""
    with tempfile.TemporaryDirectory() as tmp:
        seq_dir = Path(tmp) / "MH_TEST_seq" / "MH_TEST_seq"
        seq_dir.mkdir(parents=True)
        n = 500
        ts = np.arange(n) * 0.005  # 200 Hz
        df = pd.DataFrame({
            "timestamp": ts,
            **{c: np.sin(ts + i * 0.7) for i, c in enumerate(IMU_COLS)},
            **{c: np.cos(ts + i * 0.3) for i, c in enumerate(VEL_COLS)},
        })
        df.to_csv(seq_dir / "imu_aligned.csv", index=False)
        yield seq_dir.parent


def test_load_basic(synthetic_sequence_dir):
    seq = EuRoCSequence.load("MH_TEST_seq", synthetic_sequence_dir)
    assert seq.name == "MH_TEST_seq"
    assert seq.n_samples == 500
    assert seq.imu.shape == (500, 6)
    assert seq.gt_vel.shape == (500, 3)
    assert seq.timestamps.shape == (500,)
    assert seq.duration_s == pytest.approx(499 * 0.005, abs=1e-9)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        EuRoCSequence.load("does_not_exist", "/nonexistent/dir")


def test_outage_window_returns_indices(synthetic_sequence_dir):
    seq = EuRoCSequence.load("MH_TEST_seq", synthetic_sequence_dir)
    start, end = seq.outage_window(start_frac=0.4, duration_s=1.0)  # 200 samples @ 200 Hz
    assert 0 <= start < end < seq.n_samples
    # ~200 samples wide (allowing ±2 for searchsorted rounding)
    assert abs((end - start) - 200) <= 2


def test_load_many_skips_missing(synthetic_sequence_dir):
    out = EuRoCSequence.load_many(
        ["MH_TEST_seq", "DOES_NOT_EXIST"],
        synthetic_sequence_dir,
        skip_missing=True,
    )
    assert len(out) == 1
    assert out[0].name == "MH_TEST_seq"


def test_load_many_raises_when_skip_disabled(synthetic_sequence_dir):
    with pytest.raises(FileNotFoundError):
        EuRoCSequence.load_many(
            ["MH_TEST_seq", "DOES_NOT_EXIST"],
            synthetic_sequence_dir,
            skip_missing=False,
        )
