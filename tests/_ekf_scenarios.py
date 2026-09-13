"""Shared random EKF scenarios for the batched EKF parity/gradient tests."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from gps_denied_nav.batched_ekf import pack_states, qc_diag
from gps_denied_nav.filters.ekf import EKF15, EKFState, init_from_static, quat_to_rot

DT = 0.005


@dataclass
class Scenario:
    states: List[EKFState]   # N initial states
    imu: np.ndarray          # (T, N, 6)
    z: np.ndarray            # (T, N, 3)
    r: np.ndarray            # (T, N, 3)
    qc: np.ndarray           # (N, 12)
    mask: np.ndarray         # (T, N) bool


def make_scenario(n: int, steps: int, seed: int = 0, update_prob: float = 0.3) -> Scenario:
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        tilt = rng.normal(0.0, 0.2, 3)
        g_body = np.array([0.0, 0.0, 9.81]) + 9.81 * np.cross(tilt, [0.0, 0.0, 1.0])
        ekf = init_from_static(g_body, rng.normal(0.0, 1.0, 3))
        ekf.s.p = rng.normal(0.0, 5.0, 3)
        ekf.s.ba = rng.normal(0.0, 0.05, 3)
        ekf.s.bg = rng.normal(0.0, 0.005, 3)
        states.append(ekf.s)

    t = np.arange(steps)[:, None, None] * DT
    freq = rng.uniform(0.2, 2.0, (1, n, 6))
    phase = rng.uniform(0.0, 2 * np.pi, (1, n, 6))
    amp = np.concatenate([rng.uniform(0.2, 2.0, (1, n, 3)), rng.uniform(0.05, 1.0, (1, n, 3))], -1)
    imu = amp * np.sin(2 * np.pi * freq * t + phase)
    for i, s in enumerate(states):
        imu[:, i, 0:3] += quat_to_rot(s.q).T @ np.array([0.0, 0.0, 9.81])

    return Scenario(
        states=states,
        imu=imu,
        z=rng.normal(0.0, 1.5, (steps, n, 3)),
        r=rng.uniform(0.01, 0.3, (steps, n, 3)),
        qc=np.stack([qc_diag() * rng.uniform(0.5, 2.0) for _ in range(n)]),
        mask=rng.random((steps, n)) < update_prob,
    )


def run_numpy(sc: Scenario) -> List[List[EKFState]]:
    """Per-step EKF15 states: result[t][i] is filter i after step t."""
    filters = []
    for i, s in enumerate(sc.states):
        f = EKF15(copy.deepcopy(s))
        f._Qc = np.diag(sc.qc[i])
        filters.append(f)
    out = []
    for t in range(sc.imu.shape[0]):
        for i, f in enumerate(filters):
            f.predict(sc.imu[t, i, 0:3], sc.imu[t, i, 3:6], DT)
            if sc.mask[t, i]:
                f.update_velocity(sc.z[t, i], np.diag(sc.r[t, i]))
        out.append([copy.deepcopy(f.s) for f in filters])
    return out


def torch_inputs(
    sc: Scenario, dtype: torch.dtype, device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, P = pack_states(sc.states, dtype=dtype, device=device)

    def as_t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype, device=device)

    return x, P, as_t(sc.imu), as_t(sc.z), as_t(sc.r), as_t(sc.qc), as_t(sc.mask.astype(np.float64))
