"""Autograd integration for the custom batched EKF kernels."""
from __future__ import annotations

from typing import Any, Tuple

import torch

from . import _ext


class EKFStepFunction(torch.autograd.Function):
    """One batched EKF step with a hand-written CUDA/C++ backward.

    Only the step inputs are saved; the backward kernel recomputes the forward
    intermediates per filter, so memory per step is O(N · 241) regardless of
    how many temporaries the step uses internally.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        P: torch.Tensor,
        imu: torch.Tensor,
        z: torch.Tensor,
        r: torch.Tensor,
        qc: torch.Tensor,
        mask: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(x, P, imu, z, r, qc, mask)
        ctx.dt = dt
        x_out, P_out = _ext.load().step_forward(x, P, imu, z, r, qc, mask, dt)
        return x_out, P_out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        gx_out, gP_out = grad_outputs
        x, P, imu, z, r, qc, mask = ctx.saved_tensors
        if gx_out is None:
            gx_out = torch.zeros_like(x)
        if gP_out is None:
            gP_out = torch.zeros_like(P)
        gx, gP, gimu, gz, gr, gqc = _ext.load().step_backward(
            x, P, imu, z, r, qc, mask, gx_out.contiguous(), gP_out.contiguous(), ctx.dt)
        return gx, gP, gimu, gz, gr, gqc, None, None


def ekf_step_cuda(
    x: torch.Tensor,
    P: torch.Tensor,
    imu: torch.Tensor,
    z: torch.Tensor,
    r: torch.Tensor,
    qc: torch.Tensor,
    mask: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``reference.ekf_step`` backed by the custom kernels.

    Broadcast ``qc`` of shape ``(12,)`` is expanded to ``(N, 12)``; its gradient
    is summed back by autograd.
    """
    n = x.shape[0]
    if qc.dim() == 1:
        qc = qc.expand(n, 12)
    mask = mask.to(dtype=x.dtype)
    out: Tuple[torch.Tensor, torch.Tensor] = EKFStepFunction.apply(
        x.contiguous(), P.contiguous(), imu.contiguous(), z.contiguous(), r.contiguous(),
        qc.contiguous(), mask.contiguous(), dt)
    return out
