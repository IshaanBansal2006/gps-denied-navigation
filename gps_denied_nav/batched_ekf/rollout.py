"""Batched GPS-outage rollouts of the 15-state EKF on real sequences.

Each window: ``warmup`` samples with GPS-like ground-truth velocity updates
(attitude and biases converge), then ``outage`` samples where the only
velocity measurements come from a neural network. All windows advance in
lockstep as one batch, so every EKF step is a single kernel launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch

from ..data.euroc import EuRoCSequence
from ..filters.ekf import GRAVITY, init_from_static
from .state import pack_states

StepFn = Callable[..., Tuple[torch.Tensor, torch.Tensor]]
DT = 0.005


@dataclass
class OutageBatch:
    """Per-step inputs for N windows, time-major."""
    x0: torch.Tensor        # (N, 16)
    P0: torch.Tensor        # (N, 15, 15)
    imu: torch.Tensor       # (T, N, 6)  accel, gyro — EKF order
    z: torch.Tensor         # (T, N, 3)  GT velocity in warmup, network velocity in outage
    mask: torch.Tensor      # (T, N)     1 where a measurement arrives
    gt_vel: torch.Tensor    # (T, N, 3)  ground truth at the end of step t (sample start - warmup + t + 1)
    warmup: int
    outage: int

    @property
    def n(self) -> int:
        return int(self.x0.shape[0])


def _rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    return q / np.linalg.norm(q)


def align_initial_attitude(
    accel: np.ndarray, gyro: np.ndarray, gt_vel: np.ndarray, dt: float = DT, chunk: int = 100,
) -> np.ndarray:
    """Body→world quaternion at sample 0 from IMU + world-frame velocity (Kabsch).

    Integrates gyro to get relative attitude ΔR(t), then matches chunked
    world-frame velocity increments ``Δv - g·Δt`` to ``Σ ΔR(t) f_b(t) dt``.
    Gives full attitude including yaw, which gravity alone cannot.
    """
    n = (len(accel) // chunk) * chunk
    if n < 2 * chunk:
        raise ValueError(f"need at least {2 * chunk} samples for attitude alignment, got {len(accel)}")
    R_rel = np.eye(3)
    body, world = [], []
    acc_b = np.zeros(3)
    for i in range(n):
        acc_b += R_rel @ accel[i] * dt
        R_rel = R_rel @ _rotvec_to_matrix(gyro[i] * dt)
        if (i + 1) % chunk == 0:
            s = i + 1 - chunk
            body.append(acc_b)
            world.append(gt_vel[i + 1 if i + 1 < len(gt_vel) else i] - gt_vel[s] - GRAVITY * chunk * dt)
            acc_b = np.zeros(3)
    B = np.asarray(body)
    W = np.asarray(world)
    U, _, Vt = np.linalg.svd(W.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return _matrix_to_quat(U @ D @ Vt)


def build_outage_batch(
    seq: EuRoCSequence,
    net_vel: np.ndarray,
    starts: Sequence[int],
    warmup: int,
    outage: int,
    stride: int = 25,
    gps_stride: int = 1,
    align_samples: int = 400,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> OutageBatch:
    """Windows ``[start - warmup, start + outage)`` of one sequence.

    ``net_vel`` is the network's world-frame velocity for every sample of the
    sequence (e.g. an LSTM run causally from sample 0). Ground-truth velocity
    updates arrive every ``gps_stride`` warmup samples (1 = the EKF+GPS protocol
    of decisions 015/019); network updates every ``stride`` outage samples.
    """
    T = warmup + outage
    imu = np.zeros((T, len(starts), 6))
    z = np.zeros((T, len(starts), 3))
    mask = np.zeros((T, len(starts)))
    gt = np.zeros((T, len(starts), 3))
    states = []
    for j, start in enumerate(starts):
        a = start - warmup
        if a < 0 or start + outage + 1 > seq.n_samples:
            raise ValueError(f"{seq.name}: window start={start} needs [{a}, {start + outage}] inside [0, {seq.n_samples})")
        gyro, accel = seq.imu[a:start + outage, 0:3], seq.imu[a:start + outage, 3:6]
        imu[:, j, 0:3] = accel
        imu[:, j, 3:6] = gyro
        gt[:, j] = seq.gt_vel[a + 1:start + outage + 1]
        z[:warmup, j] = gt[:warmup, j]
        z[warmup:, j] = net_vel[start:start + outage]
        mask[gps_stride - 1:warmup:gps_stride, j] = 1.0
        mask[warmup + stride - 1::stride, j] = 1.0

        ekf = init_from_static(accel[0].astype(np.float64), seq.gt_vel[a].astype(np.float64))
        ekf.s.q = align_initial_attitude(accel[:align_samples], gyro[:align_samples], seq.gt_vel[a:a + align_samples + 1])
        states.append(ekf.s)

    x0, P0 = pack_states(states, dtype=dtype, device=device)

    def as_t(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return OutageBatch(x0, P0, as_t(imu), as_t(z), as_t(mask), as_t(gt), warmup, outage)


def rollout(
    step: StepFn,
    batch: OutageBatch,
    qc: torch.Tensor,
    r_gps: torch.Tensor,
    r_net: torch.Tensor,
    grad_warmup: bool = False,
) -> torch.Tensor:
    """Velocity estimates during the outage, shape ``(outage, N, 3)``.

    ``qc`` is (12,) or (N, 12); ``r_gps``/``r_net`` are (3,) variances. The
    warmup runs without autograd unless ``grad_warmup``.
    """
    n = batch.n
    x, P = batch.x0, batch.P0
    r_gps_n = r_gps.expand(n, 3)
    r_net_n = r_net.expand(n, 3)
    qc = qc.expand(n, 12)
    with torch.set_grad_enabled(grad_warmup and torch.is_grad_enabled()):
        for t in range(batch.warmup):
            x, P = step(x, P, batch.imu[t], batch.z[t], r_gps_n, qc, batch.mask[t], DT)
    if not grad_warmup:
        x, P = x.detach(), P.detach()
    vels = []
    for t in range(batch.warmup, batch.warmup + batch.outage):
        x, P = step(x, P, batch.imu[t], batch.z[t], r_net_n, qc, batch.mask[t], DT)
        vels.append(x[:, 3:6])
    return torch.stack(vels)
