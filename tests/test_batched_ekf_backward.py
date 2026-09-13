"""Hand-written backward kernels vs PyTorch autograd through the reference.

Skipped when no GPU or no nvcc toolchain is available (e.g. CI).
"""
from __future__ import annotations

import pytest
import torch

from gps_denied_nav.batched_ekf import _ext, ekf_step
from gps_denied_nav.batched_ekf.function import ekf_step_cuda
from tests._ekf_scenarios import DT, make_scenario, torch_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _ext.is_available(),
    reason="batched EKF CUDA extension needs a GPU and nvcc (scripts/setup_cuda_toolchain.sh)",
)

INPUT_NAMES = ("x", "P", "imu", "z", "r", "qc")


def _grads(step, device, dtype, n=32, steps=1, seed=7, update_prob=0.5):
    torch.manual_seed(seed)
    sc = make_scenario(n=n, steps=steps, seed=seed, update_prob=update_prob)
    x, P, imu, z, r, qc, mask = torch_inputs(sc, dtype, device)
    leaves = [t.clone().requires_grad_(True) for t in (x, P, imu, z, r, qc)]
    xs, Ps, imu_l, z_l, r_l, qc_l = leaves
    for t in range(steps):
        xs, Ps = step(xs, Ps, imu_l[t], z_l[t], r_l[t], qc_l, mask[t], DT)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    wx = torch.randn(xs.shape, generator=gen, dtype=torch.float64).to(device, dtype)
    wP = torch.randn(Ps.shape, generator=gen, dtype=torch.float64).to(device, dtype)
    loss = (xs * wx).sum() + (Ps * wP).sum()
    return torch.autograd.grad(loss, leaves)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_single_step_grads_match_autograd(device):
    ref = _grads(ekf_step, device, torch.float64)
    ext = _grads(ekf_step_cuda, device, torch.float64)
    for name, a, b in zip(INPUT_NAMES, ext, ref):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10, msg=lambda m: f"grad {name}: {m}")


def test_multi_step_grads_match_autograd():
    ref = _grads(ekf_step, "cuda", torch.float64, n=16, steps=25, seed=8)
    ext = _grads(ekf_step_cuda, "cuda", torch.float64, n=16, steps=25, seed=8)
    for name, a, b in zip(INPUT_NAMES, ext, ref):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9, msg=lambda m: f"grad {name}: {m}")


def test_all_update_and_no_update_branches():
    for prob in (0.0, 1.0):
        ref = _grads(ekf_step, "cuda", torch.float64, n=8, seed=9, update_prob=prob)
        ext = _grads(ekf_step_cuda, "cuda", torch.float64, n=8, seed=9, update_prob=prob)
        for name, a, b in zip(INPUT_NAMES, ext, ref):
            torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10, msg=lambda m: f"p={prob} grad {name}: {m}")


def test_float32_grads_track_float64():
    g64 = _grads(ekf_step_cuda, "cuda", torch.float64, n=32, steps=5, seed=10)
    g32 = _grads(ekf_step_cuda, "cuda", torch.float32, n=32, steps=5, seed=10)
    for name, a, b in zip(INPUT_NAMES, g32, g64):
        scale = b.abs().max().clamp_min(1.0)
        err = (a.double() - b).abs().max() / scale
        assert err < 1e-4, f"float32 grad {name} relative error {err:.2e}"


def test_extension_gradcheck():
    sc = make_scenario(n=2, steps=1, seed=11, update_prob=1.0)
    x, P, imu, z, r, qc, _ = torch_inputs(sc, torch.float64, "cuda")
    mask = torch.tensor([1.0, 0.0], dtype=torch.float64, device="cuda")

    def fn(x_, P_, imu_, z_, r_, qc_):
        return ekf_step_cuda(x_, P_, imu_, z_, r_, qc_, mask, DT)

    inputs = tuple(t.clone().requires_grad_(True) for t in (x, P, imu[0], z[0], r[0], qc))
    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)


def test_broadcast_qc_gradient_sums_over_filters():
    sc = make_scenario(n=6, steps=1, seed=12)
    x, P, imu, z, r, _, mask = torch_inputs(sc, torch.float64, "cuda")
    qc = torch.full((12,), 1e-3, dtype=torch.float64, device="cuda", requires_grad=True)
    qc_ref = qc.detach().clone().requires_grad_(True)
    ekf_step_cuda(x, P, imu[0], z[0], r[0], qc, mask[0], DT)[1].sum().backward()
    ekf_step(x, P, imu[0], z[0], r[0], qc_ref.expand(6, 12), mask[0], DT)[1].sum().backward()
    torch.testing.assert_close(qc.grad, qc_ref.grad)
