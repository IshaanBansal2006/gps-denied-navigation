"""Outage rollout helpers: attitude alignment and batch construction (CPU)."""
from __future__ import annotations

import numpy as np
import torch

from gps_denied_nav.batched_ekf import ekf_step, qc_diag
from gps_denied_nav.batched_ekf.rollout import DT, align_initial_attitude, build_outage_batch, rollout
from gps_denied_nav.data.euroc import EuRoCSequence
from gps_denied_nav.filters.ekf import GRAVITY, quat_to_rot


def _synthetic_sequence(n: int, R0: np.ndarray, seed: int = 0) -> EuRoCSequence:
    """Constant body rates and a known world-frame acceleration profile."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    acc_w = np.stack([np.sin(0.7 * t), np.cos(0.4 * t), 0.3 * np.sin(1.1 * t)], axis=1)
    gyro = np.tile(rng.normal(0.0, 0.2, 3), (n, 1))
    R = R0.copy()
    accel = np.zeros((n, 3))
    for i in range(n):
        accel[i] = R.T @ (acc_w[i] - GRAVITY)
        w = gyro[i] * DT
        th = np.linalg.norm(w)
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = R @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
    vel = np.vstack([np.zeros(3), np.cumsum(acc_w * DT, axis=0)])[:n]
    return EuRoCSequence(
        name="synthetic",
        timestamps=t,
        imu=np.hstack([gyro, accel]).astype(np.float32),
        gt_vel=vel.astype(np.float32),
    )


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_align_initial_attitude_recovers_yaw():
    R0 = _rot_z(2.1) @ _rot_z(0.0)
    seq = _synthetic_sequence(800, R0)
    q = align_initial_attitude(seq.imu[:, 3:6].astype(np.float64), seq.imu[:, 0:3].astype(np.float64),
                               seq.gt_vel.astype(np.float64))
    angle = np.degrees(np.arccos(np.clip((np.trace(quat_to_rot(q).T @ R0) - 1) / 2, -1, 1)))
    assert angle < 2.0, f"attitude alignment off by {angle:.2f} deg"


def test_build_outage_batch_layout_and_rollout_shapes():
    seq = _synthetic_sequence(1200, _rot_z(-0.7))
    net_vel = seq.gt_vel + 0.1
    batch = build_outage_batch(seq, net_vel, starts=[500, 600], warmup=400, outage=300, stride=25, gps_stride=8)
    assert batch.imu.shape == (700, 2, 6)
    np.testing.assert_allclose(batch.imu[:, 0, 0:3].numpy(), seq.imu[100:800, 3:6], rtol=1e-6)
    np.testing.assert_allclose(batch.gt_vel[:, 1].numpy(), seq.gt_vel[201:901], rtol=1e-6)
    np.testing.assert_allclose(batch.z[400:, 0].numpy(), net_vel[500:800], rtol=1e-6)
    assert batch.mask[:400, 0].sum().item() == 400 // 8
    assert batch.mask[400:, 0].sum().item() == 300 // 25
    assert batch.mask[424, 0].item() == 1.0

    qc = torch.as_tensor(qc_diag(), dtype=torch.float64)
    r = torch.full((3,), 0.1, dtype=torch.float64)
    vel = rollout(ekf_step, batch, qc, r * 0.01, r)
    assert vel.shape == (300, 2, 3)
    err = torch.linalg.norm(vel - batch.gt_vel[400:], dim=-1)
    assert err.max().item() < 0.5, "EKF on noiseless synthetic data with 0.1 m/s biased measurements should stay close"
