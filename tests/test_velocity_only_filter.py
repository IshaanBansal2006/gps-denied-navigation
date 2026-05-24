"""Unit tests for gps_denied_nav.filters.VelocityOnlyFilter."""
from __future__ import annotations

import numpy as np

from gps_denied_nav.filters import VelocityOnlyFilter


def test_reset_initializes_state():
    f = VelocityOnlyFilter()
    f.reset(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_almost_equal(f.velocity, [1.0, 2.0, 3.0])


def test_predict_grows_covariance():
    f = VelocityOnlyFilter(sigma_process=0.5)
    f.reset(np.zeros(3))
    P0 = f.P.copy()
    f.predict(dt=0.005)
    assert (f.P >= P0).all(), "predict should not decrease covariance"
    assert (f.P > P0).any(), "predict should grow covariance"
    # Velocity unchanged in pure prediction step.
    np.testing.assert_array_almost_equal(f.velocity, np.zeros(3))


def test_update_pulls_estimate_toward_measurement():
    f = VelocityOnlyFilter()
    f.reset(np.zeros(3))
    f.predict(dt=0.005)
    z = np.array([1.0, -2.0, 0.5])
    f.update(z)
    # The Kalman gain mixes; we should land between 0 and z (exclusive).
    for i in range(3):
        if z[i] > 0:
            assert 0 < f.velocity[i] < z[i] + 1e-6
        elif z[i] < 0:
            assert z[i] - 1e-6 < f.velocity[i] < 0
        else:
            assert abs(f.velocity[i]) < 1e-6


def test_repeated_updates_converge_to_measurement():
    """Many measurements at the same velocity should shrink P toward zero and pull v toward z."""
    f = VelocityOnlyFilter()
    f.reset(np.zeros(3))
    z = np.array([3.0, 0.0, -1.5])
    for _ in range(200):
        f.predict(0.005)
        f.update(z)
    np.testing.assert_allclose(f.velocity, z, atol=0.05)
    # Covariance should have shrunk substantially.
    assert np.trace(f.P) < 1.0


def test_custom_R_diag_changes_kalman_gain_magnitude():
    """Smaller R_diag => trust measurements more => velocity moves further per update."""
    z = np.array([10.0, 0.0, 0.0])

    f_trust = VelocityOnlyFilter(R_diag=(0.01, 0.01, 0.01))
    f_dist = VelocityOnlyFilter(R_diag=(10.0, 10.0, 10.0))
    for f in (f_trust, f_dist):
        f.reset(np.zeros(3))
        f.predict(0.005)
        f.update(z)
    assert f_trust.velocity[0] > f_dist.velocity[0], \
        "lower R should produce larger step toward measurement"
