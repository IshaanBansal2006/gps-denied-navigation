"""Conversions between the NumPy ``EKF15`` and the packed batched layout."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from ..filters.ekf import EKF15, EKFState


def qc_diag(
    sigma_a: float = 0.05,
    sigma_g: float = 0.005,
    sigma_ba: float = 5e-4,
    sigma_bg: float = 5e-5,
) -> np.ndarray:
    """Continuous IMU noise variances in the order EKF15 builds ``Qc``."""
    return np.array([sigma_a] * 3 + [sigma_g] * 3 + [sigma_ba] * 3 + [sigma_bg] * 3) ** 2


def pack_states(
    states: Sequence[EKFState],
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack EKF15 states into ``x (N, 16)`` and ``P (N, 15, 15)``."""
    x = np.stack([np.concatenate([s.p, s.v, s.q, s.ba, s.bg]) for s in states])
    P = np.stack([s.P for s in states])
    return (
        torch.as_tensor(x, dtype=dtype, device=device),
        torch.as_tensor(P, dtype=dtype, device=device),
    )


def unpack_state(x: torch.Tensor, P: torch.Tensor, i: int) -> EKFState:
    xi = x[i].detach().cpu().double().numpy()
    return EKFState(
        p=xi[0:3].copy(), v=xi[3:6].copy(), q=xi[6:10].copy(),
        ba=xi[10:13].copy(), bg=xi[13:16].copy(),
        P=P[i].detach().cpu().double().numpy().copy(),
    )


def qc_from_ekf(ekf: EKF15) -> np.ndarray:
    return np.diag(ekf._Qc).copy()
