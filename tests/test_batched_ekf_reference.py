"""Batched PyTorch EKF reference vs the NumPy EKF15, plus autograd sanity."""
from __future__ import annotations

import numpy as np
import torch

from gps_denied_nav.batched_ekf import ekf_step
from gps_denied_nav.batched_ekf.reference import rotvec_to_quat
from gps_denied_nav.filters.ekf import quat_from_rotvec
from tests._ekf_scenarios import DT, make_scenario, run_numpy, torch_inputs


def test_rotvec_to_quat_matches_numpy_across_taylor_switch():
    for theta in [1e-9, 1e-5, 0.01, 0.049, 0.051, 0.3, 2.0]:
        axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
        rv = axis * theta
        got = rotvec_to_quat(torch.tensor(rv[None], dtype=torch.float64))[0].numpy()
        np.testing.assert_allclose(got, quat_from_rotvec(rv), atol=1e-10)


def test_reference_matches_numpy_ekf15_over_trajectory():
    sc = make_scenario(n=6, steps=300, seed=1)
    expected = run_numpy(sc)
    x, P, imu, z, r, qc, mask = torch_inputs(sc, torch.float64, "cpu")
    for t in range(imu.shape[0]):
        x, P = ekf_step(x, P, imu[t], z[t], r[t], qc, mask[t], DT)
        if t % 50 == 49 or t == imu.shape[0] - 1:
            for i, s in enumerate(expected[t]):
                np.testing.assert_allclose(
                    x[i].numpy(), np.concatenate([s.p, s.v, s.q, s.ba, s.bg]), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(P[i].numpy(), s.P, rtol=1e-8, atol=1e-10)


def test_reference_step_passes_gradcheck():
    sc = make_scenario(n=3, steps=1, seed=2, update_prob=1.0)
    x, P, imu, z, r, qc, _ = torch_inputs(sc, torch.float64, "cpu")
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)

    def fn(x_, P_, imu_, z_, r_, qc_):
        return ekf_step(x_, P_, imu_, z_, r_, qc_, mask, DT)

    inputs = tuple(t.clone().requires_grad_(True) for t in (x, P, imu[0], z[0], r[0], qc))
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)
