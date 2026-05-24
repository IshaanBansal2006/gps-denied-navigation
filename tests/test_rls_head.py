"""Unit tests for gps_denied_nav.adaptation.RLSHead."""
from __future__ import annotations

import numpy as np
import pytest

from gps_denied_nav.adaptation import RLSHead


def test_reset_loads_pretrained_weights():
    """reset() should copy weights from torch nn.Linear convention."""
    in_dim, out_dim = 8, 3
    head = RLSHead(in_dim=in_dim, out_dim=out_dim)
    W0 = np.random.randn(in_dim, out_dim).astype(np.float32)
    b0 = np.random.randn(out_dim).astype(np.float32)
    head.reset(W0, b0)
    np.testing.assert_array_almost_equal(head.W, W0.astype(np.float64))
    np.testing.assert_array_almost_equal(head.b, b0.astype(np.float64))
    assert head.n_updates == 0


def test_predict_matches_linear_layer_after_reset():
    """Right after reset(), predict(x) should equal W^T x + b (i.e., the pre-trained head)."""
    in_dim, out_dim = 4, 2
    head = RLSHead(in_dim=in_dim, out_dim=out_dim)
    W0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-1.0, 2.0]], dtype=np.float32)
    b0 = np.array([0.1, -0.2], dtype=np.float32)
    head.reset(W0, b0)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pred = head.predict(x)
    expected = W0.T @ x + b0
    np.testing.assert_allclose(pred, expected, rtol=1e-5)


def test_update_reduces_residual_on_repeated_pair():
    """Repeatedly updating with the same (x, y) should pull the prediction toward y."""
    in_dim, out_dim = 5, 2
    head = RLSHead(in_dim=in_dim, out_dim=out_dim, p_init=10.0, forgetting=0.99)
    head.reset(
        np.zeros((in_dim, out_dim), dtype=np.float32),
        np.zeros(out_dim, dtype=np.float32),
    )
    x = np.array([1.0, 0.0, 0.5, -0.2, 0.1], dtype=np.float32)
    y = np.array([2.0, -1.0], dtype=np.float32)

    initial_err = np.linalg.norm(head.predict(x) - y)
    for _ in range(20):
        head.update(x, y)
    final_err = np.linalg.norm(head.predict(x) - y)
    assert final_err < initial_err * 0.5, \
        f"RLS should reduce residual; initial={initial_err}, final={final_err}"


def test_n_updates_increments():
    head = RLSHead(in_dim=3, out_dim=2)
    head.reset(np.zeros((3, 2), dtype=np.float32), np.zeros(2, dtype=np.float32))
    for i in range(5):
        head.update(np.ones(3, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32))
    assert head.n_updates == 5


def test_dimensions_assert():
    head = RLSHead(in_dim=4, out_dim=2)
    head.reset(np.zeros((4, 2), dtype=np.float32), np.zeros(2, dtype=np.float32))
    with pytest.raises(AssertionError):
        head.update(np.zeros(3, dtype=np.float32),
                    np.zeros(2, dtype=np.float32))  # wrong x dim
    with pytest.raises(AssertionError):
        head.update(np.zeros(4, dtype=np.float32),
                    np.zeros(3, dtype=np.float32))  # wrong y dim


def test_forgetting_factor_decays_old_information():
    """With aggressive forgetting (λ small), recent samples should dominate the fit."""
    in_dim, out_dim = 2, 1
    head = RLSHead(in_dim=in_dim, out_dim=out_dim, forgetting=0.5, p_init=10.0)
    head.reset(np.zeros((in_dim, out_dim), dtype=np.float32),
               np.zeros(out_dim, dtype=np.float32))
    x = np.array([1.0, 0.0], dtype=np.float32)
    # Feed many old samples mapping x → 5
    for _ in range(50):
        head.update(x, np.array([5.0], dtype=np.float32))
    pred_old = head.predict(x)[0]
    # Then feed a few recent samples mapping x → -10
    for _ in range(50):
        head.update(x, np.array([-10.0], dtype=np.float32))
    pred_new = head.predict(x)[0]
    # New prediction should have moved far from old target toward new one
    assert pred_new < pred_old, \
        f"With forgetting={head.lam}, new evidence should dominate. " \
        f"pred_old={pred_old}, pred_new={pred_new}"
