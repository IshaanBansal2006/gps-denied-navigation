"""Custom C++/CUDA batched EKF kernels vs the PyTorch reference.

Skipped when no GPU or no nvcc toolchain is available (e.g. CI).
"""
from __future__ import annotations

import pytest
import torch

from gps_denied_nav.batched_ekf import _ext, ekf_step
from tests._ekf_scenarios import DT, make_scenario, torch_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext.is_available(),
    reason="batched EKF CUDA extension needs a GPU and nvcc (scripts/setup_cuda_toolchain.sh)",
)


def _rollout(step, x, P, imu, z, r, qc, mask):
    for t in range(imu.shape[0]):
        x, P = step(x, P, imu[t].contiguous(), z[t].contiguous(), r[t].contiguous(), qc, mask[t].contiguous(), DT)
    return x, P


def _ext_step(x, P, imu, z, r, qc, mask, dt):
    return _ext.load().step_forward(x, P, imu, z, r, qc, mask, dt)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_forward_matches_reference_float64(device):
    sc = make_scenario(n=64, steps=200, seed=3)
    inputs = torch_inputs(sc, torch.float64, device)
    x_ref, P_ref = _rollout(ekf_step, *inputs)
    x_ext, P_ext = _rollout(_ext_step, *inputs)
    torch.testing.assert_close(x_ext, x_ref, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(P_ext, P_ref, rtol=1e-9, atol=1e-11)


def test_forward_float32_tracks_float64():
    sc = make_scenario(n=64, steps=200, seed=4)
    x64, P64 = _rollout(_ext_step, *torch_inputs(sc, torch.float64, "cuda"))
    x32, P32 = _rollout(_ext_step, *torch_inputs(sc, torch.float32, "cuda"))
    torch.testing.assert_close(x32.double(), x64, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(P32.double(), P64, rtol=1e-2, atol=1e-4)


def test_predict_only_rows_skip_update():
    sc = make_scenario(n=8, steps=1, seed=5, update_prob=0.0)
    x, P, imu, z, r, qc, mask = torch_inputs(sc, torch.float64, "cuda")
    z_far = z + 100.0
    a = _ext_step(x, P, imu[0], z[0], r[0], qc, mask[0], DT)
    b = _ext_step(x, P, imu[0], z_far[0], r[0], qc, mask[0], DT)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_rejects_mismatched_inputs():
    sc = make_scenario(n=4, steps=1, seed=6)
    x, P, imu, z, r, qc, mask = torch_inputs(sc, torch.float64, "cuda")
    with pytest.raises(RuntimeError, match="same device"):
        _ext_step(x, P, imu[0].cpu(), z[0], r[0], qc, mask[0], DT)
    with pytest.raises(RuntimeError, match="dtype"):
        _ext_step(x, P.float(), imu[0], z[0], r[0], qc, mask[0], DT)
    with pytest.raises(RuntimeError, match="shape"):
        _ext_step(x, P, imu[0, :3], z[0], r[0], qc, mask[0], DT)
