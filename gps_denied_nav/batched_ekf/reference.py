"""Batched, differentiable 15-state error-state EKF in pure PyTorch.

This is the ground-truth implementation for the custom CUDA kernels: the
forward pass matches ``gps_denied_nav.filters.EKF15`` step-for-step, and
autograd through it defines the gradients the hand-written backward kernel
must reproduce.

Packed layout (N filters):
    x   (N, 16)      nominal state  [p(3), v(3), q(4) wxyz, ba(3), bg(3)]
    P   (N, 15, 15)  error-state covariance [δp, δv, δφ, δba, δbg]
    imu (N, 6)       [accel(3), gyro(3)] body frame
    z   (N, 3)       world-frame velocity measurement
    r   (N, 3)       measurement noise variances (diagonal R)
    qc  (N, 12)      continuous IMU noise variances [accel, gyro, ba-rw, bg-rw]
    mask (N,)        1 → apply the velocity update after predict, 0 → predict only
"""
from __future__ import annotations

from typing import Tuple

import torch

GRAVITY_Z = -9.81
STATE_DIM = 16
ERR_DIM = 15

# θ below which sin/cos in the rotation-vector map switch to a Taylor series.
# Keeps the map and its derivative smooth and precise in float32.
SMALL_ANGLE_SQ = 2.5e-3


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.sqrt((q * q).sum(-1, keepdim=True))


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([
        torch.stack([o, -z, y], -1),
        torch.stack([z, o, -x], -1),
        torch.stack([-y, x, o], -1),
    ], dim=-2)


def rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """Rotation vector → unit quaternion, smooth through θ = 0."""
    t2 = (rv * rv).sum(-1, keepdim=True)
    small = t2 < SMALL_ANGLE_SQ
    t2_safe = torch.where(small, torch.ones_like(t2), t2)
    theta = torch.sqrt(t2_safe)
    w_big = torch.cos(0.5 * theta)
    s_big = torch.sin(0.5 * theta) / theta
    w_small = 1 - t2 / 8 + t2 * t2 / 384 - t2 * t2 * t2 / 46080
    s_small = 0.5 - t2 / 48 + t2 * t2 / 3840 - t2 * t2 * t2 / 645120
    w = torch.where(small, w_small, w_big)
    s = torch.where(small, s_small, s_big)
    return torch.cat([w, s * rv], dim=-1)


def _eye(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, dtype=like.dtype, device=like.device)


def predict(
    x: torch.Tensor, P: torch.Tensor, imu: torch.Tensor, qc: torch.Tensor, dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    p, v, q, ba, bg = x.split([3, 3, 4, 3, 3], dim=-1)
    accel, gyro = imu.split([3, 3], dim=-1)

    R = quat_to_rot(q)
    a_c = accel - ba
    w_c = gyro - bg
    gravity = torch.zeros_like(a_c)
    gravity[:, 2] = GRAVITY_Z
    f = (R @ a_c.unsqueeze(-1)).squeeze(-1) + gravity

    v1 = v + f * dt
    p1 = p + v * dt + 0.5 * f * dt * dt
    q1 = quat_normalize(quat_mul(q, rotvec_to_quat(w_c * dt)))

    F = x.new_zeros(n, ERR_DIM, ERR_DIM)
    F[:, 0:3, 3:6] = _eye(3, x)
    F[:, 3:6, 6:9] = -R @ skew(a_c)
    F[:, 3:6, 9:12] = -R
    F[:, 6:9, 6:9] = -skew(w_c)
    F[:, 6:9, 12:15] = -_eye(3, x)
    Phi = _eye(ERR_DIM, x) + F * dt

    G = x.new_zeros(n, ERR_DIM, 12)
    G[:, 3:6, 0:3] = -R
    G[:, 6:9, 3:6] = -_eye(3, x)
    G[:, 9:12, 6:9] = _eye(3, x)
    G[:, 12:15, 9:12] = _eye(3, x)
    Qd = G @ torch.diag_embed(qc) @ G.transpose(-1, -2) * dt

    P1 = Phi @ P @ Phi.transpose(-1, -2) + Qd
    return torch.cat([p1, v1, q1, ba, bg], dim=-1), P1


def update_velocity(
    x: torch.Tensor, P: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    p, v, q, ba, bg = x.split([3, 3, 4, 3, 3], dim=-1)
    Rm = torch.diag_embed(r)

    S = P[:, 3:6, 3:6] + Rm
    K = P[:, :, 3:6] @ torch.linalg.inv(S)
    dx = (K @ (z - v).unsqueeze(-1)).squeeze(-1)

    q2 = quat_normalize(quat_mul(rotvec_to_quat(dx[:, 6:9]), q))
    x2 = torch.cat([p + dx[:, 0:3], v + dx[:, 3:6], q2, ba + dx[:, 9:12], bg + dx[:, 12:15]], dim=-1)

    KH = x.new_zeros(n, ERR_DIM, ERR_DIM)
    KH[:, :, 3:6] = K
    A = _eye(ERR_DIM, x) - KH
    P2 = A @ P @ A.transpose(-1, -2) + K @ Rm @ K.transpose(-1, -2)
    return x2, P2


def ekf_step(
    x: torch.Tensor,
    P: torch.Tensor,
    imu: torch.Tensor,
    z: torch.Tensor,
    r: torch.Tensor,
    qc: torch.Tensor,
    mask: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One predict step, followed by a velocity update where ``mask`` is set."""
    x1, P1 = predict(x, P, imu, qc, dt)
    x2, P2 = update_velocity(x1, P1, z, r)
    m = mask.to(dtype=torch.bool)
    return (
        torch.where(m[:, None], x2, x1),
        torch.where(m[:, None, None], P2, P1),
    )
