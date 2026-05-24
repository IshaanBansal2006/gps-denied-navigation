"""Unit tests for gps_denied_nav.adaptation.ContinuousAdapter."""
from __future__ import annotations

import numpy as np
import pytest

from gps_denied_nav.adaptation import ContinuousAdapter, RLSHead


@pytest.fixture
def adapter():
    rls = RLSHead(in_dim=4, out_dim=3, forgetting=0.995, p_init=0.1)
    rls.reset(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    return ContinuousAdapter(rls=rls, alpha_smooth=0.0, ema_alpha=0.95, outage_lambda=0.999)


def test_reset_overrides_rls_lambda(adapter):
    """reset() should swap the inner RLS lambda to outage_lambda; restore() puts it back."""
    original_lam = adapter.rls.lam
    adapter.reset(
        v_init_norm=np.zeros(3),
        v_init_world=np.zeros(3),
    )
    assert adapter.rls.lam == pytest.approx(adapter.outage_lambda)
    adapter.restore()
    assert adapter.rls.lam == pytest.approx(original_lam)


def test_update_during_outage_calls_rls(adapter):
    """A single update should increment RLS n_updates."""
    norm = {
        "y_mean": np.zeros(3, dtype=np.float32),
        "y_std":  np.ones(3, dtype=np.float32),
    }
    adapter.reset(v_init_norm=np.zeros(3), v_init_world=np.zeros(3))
    initial_n = adapter.rls.n_updates
    adapter.update_during_outage(
        h=np.array([1.0, 0.0, 0.5, -0.2], dtype=np.float64),
        v_filter_world=np.array([1.0, 0.5, 0.0]),
        gyro=np.array([0.0, 0.0, 0.1]),
        dt=0.005,
        norm=norm,
    )
    assert adapter.rls.n_updates == initial_n + 1
    adapter.restore()


def test_predict_delegates_to_rls(adapter):
    h = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(adapter.predict(h), adapter.rls.predict(h))
