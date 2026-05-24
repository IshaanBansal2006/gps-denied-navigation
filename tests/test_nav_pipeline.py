"""Unit tests for gps_denied_nav.NavPipeline (end-to-end with a tiny synthetic LSTM).

These tests exercise the *plumbing* — that warmup, outage prediction, filter
updates, and adapter calls all wire together. They do NOT load real EuRoC
data or check actual model accuracy; that's covered by the eval scripts.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gps_denied_nav import NavPipeline, OutageEvaluator, OutageResult
from gps_denied_nav.adaptation import RLSHead
from gps_denied_nav.data import IMU_COLS, VEL_COLS, EuRoCSequence
from gps_denied_nav.filters import VelocityOnlyFilter
from gps_denied_nav.models import LSTMRegressor


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return LSTMRegressor(input_size=6, hidden_size=8, num_layers=1,
                          output_dim=3, dropout=0.0)


@pytest.fixture
def tiny_norm():
    return {
        "x_mean": np.zeros(6, dtype=np.float32),
        "x_std":  np.ones(6, dtype=np.float32),
        "y_mean": np.zeros(3, dtype=np.float32),
        "y_std":  np.ones(3, dtype=np.float32),
    }


@pytest.fixture
def tiny_sequence():
    """200-sample synthetic sequence — long enough for a short outage."""
    n = 600
    ts = np.arange(n) * 0.005
    rng = np.random.default_rng(0)
    return EuRoCSequence(
        name="synthetic",
        timestamps=ts.astype(np.float64),
        imu=rng.normal(0, 0.5, size=(n, 6)).astype(np.float32),
        gt_vel=np.column_stack([
            np.sin(ts), np.cos(ts), 0.1 * ts,
        ]).astype(np.float32),
    )


def test_pipeline_runs_without_adapter(tiny_model, tiny_norm, tiny_sequence):
    device = torch.device("cpu")
    pipeline = NavPipeline(model=tiny_model, adapter=None,
                            filter=VelocityOnlyFilter(),
                            norm=tiny_norm, device=device, update_stride=25)
    result = pipeline.run_outage(tiny_sequence, outage_start=200, outage_end=400)
    assert isinstance(result, OutageResult)
    assert result.velocity_estimate.shape == (201, 3)
    assert result.velocity_gt.shape == (201, 3)
    assert result.position_estimate.shape == (201, 3)
    # Position at t=0 should be the origin (we integrate from there).
    np.testing.assert_array_equal(result.position_estimate[0], np.zeros(3))


def test_pipeline_runs_with_rls_adapter(tiny_model, tiny_norm, tiny_sequence):
    """Adapter should be called during warmup and predicted from during outage."""
    device = torch.device("cpu")
    adapter = RLSHead(in_dim=8, out_dim=3, forgetting=0.995, p_init=0.1)
    pipeline = NavPipeline(model=tiny_model, adapter=adapter,
                            filter=VelocityOnlyFilter(),
                            norm=tiny_norm, device=device, update_stride=25)
    # The pipeline ctor should have called adapter.reset() with the model head weights.
    assert pipeline.adapter is adapter
    result = pipeline.run_outage(tiny_sequence, outage_start=200, outage_end=400)
    # Warmup should have happened — n_updates > 0.
    assert adapter.n_updates > 0
    assert result.velocity_estimate.shape == (201, 3)


def test_outage_evaluator_metrics_finite(tiny_model, tiny_norm, tiny_sequence):
    device = torch.device("cpu")
    pipeline = NavPipeline(model=tiny_model, adapter=None,
                            filter=VelocityOnlyFilter(),
                            norm=tiny_norm, device=device, update_stride=25)
    ev = OutageEvaluator(tiny_sequence, outage_start_frac=0.4)
    _, metrics = ev.evaluate(pipeline, outage_duration_s=0.5)
    for v in metrics.as_dict().values():
        assert np.isfinite(v), f"non-finite metric: {metrics}"


def test_pipeline_result_dataclass_arithmetic(tiny_model, tiny_norm, tiny_sequence):
    """OutageResult.{velocity_error, position_error, final_error_norm} should be consistent."""
    device = torch.device("cpu")
    pipeline = NavPipeline(model=tiny_model, adapter=None,
                            filter=VelocityOnlyFilter(),
                            norm=tiny_norm, device=device, update_stride=25)
    result = pipeline.run_outage(tiny_sequence, outage_start=200, outage_end=400)
    # final_error_norm should equal the last entry of velocity_error.
    assert result.final_error_norm == pytest.approx(result.velocity_error[-1])
    # mean_error_norm should equal the mean.
    assert result.mean_error_norm == pytest.approx(result.velocity_error.mean(), rel=1e-6)
