"""
15-state error-state EKF for strapdown INS navigation.

Nominal state: p(3) + v(3) + q(4) + ba(3) + bg(3) = 16 values
Error state:   δp(3) + δv(3) + δφ(3) + δba(3) + δbg(3) = 15 values

World frame: ENU (x-east, y-north, z-up), g = [0, 0, -9.81] m/s²
Quaternion convention: [w, x, y, z], rotation from body → world
    v_world = R(q) @ v_body
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


# --- Quaternion utilities ---------------------------------------------------

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q, [w,x,y,z] convention."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] → 3×3 rotation matrix (body → world)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.linalg.norm(axis)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / n])


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    """Small rotation vector → quaternion."""
    angle = np.linalg.norm(rv)
    return quat_from_axis_angle(rv, angle)


def quat_from_gravity(g_body: np.ndarray) -> np.ndarray:
    """
    Initialize quaternion from static accelerometer reading.
    Finds rotation (body→world) such that R(q) @ g_body ≈ [0, 0, -9.81].
    Yaw is set to 0 (unobservable from accelerometer alone).
    """
    g_hat = g_body / np.linalg.norm(g_body)      # should ≈ R(q)^T @ [0,0,-1] * (-1)
    # specific force at rest = R(q)^T @ (-g_world) = R(q)^T @ [0,0,9.81]
    # so g_hat = R(q)^T @ [0,0,1]   (column 3 of R^T = row 3 of R)
    # We need R such that the third row of R = g_hat^T

    # Align [0,0,1]_body with g_hat direction in body frame using Rodrigues' formula
    # v_from = [0,0,1], v_to = g_hat (unit vector in body frame representing world-up)
    # We need R_bw s.t. R_bw @ g_hat = [0,0,1] (gravity maps to world-up).
    # This is the rotation that takes g_hat → [0,0,1].
    v_from = g_hat
    v_to = np.array([0.0, 0.0, 1.0])
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if dot < -0.9999:
        # 180° rotation — pick any perpendicular axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return quat_from_axis_angle(axis, np.pi)
    s = np.linalg.norm(cross)
    if s < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    # Rodrigues' formula for quaternion from two vectors
    angle = np.arctan2(s, dot)
    return quat_from_axis_angle(cross / s, angle)


# --- Skew symmetric matrix --------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# --- EKF --------------------------------------------------------------------

@dataclass
class EKFState:
    p: np.ndarray       # position (3)
    v: np.ndarray       # velocity (3)
    q: np.ndarray       # quaternion body→world [w,x,y,z] (4)
    ba: np.ndarray      # accelerometer bias (3)
    bg: np.ndarray      # gyroscope bias (3)
    P: np.ndarray       # error-state covariance (15×15)


class EKF15:
    """
    15-state strapdown INS EKF (error-state formulation).

    Noise parameters (tuning):
        sigma_a    : accelerometer white noise (m/s²)
        sigma_g    : gyroscope white noise (rad/s)
        sigma_ba   : accel bias random-walk std (m/s² / √s)
        sigma_bg   : gyro bias random-walk std (rad/s / √s)
        sigma_v    : velocity measurement noise std (m/s)
    """

    def __init__(
        self,
        state: EKFState,
        sigma_a: float = 0.05,
        sigma_g: float = 0.005,
        sigma_ba: float = 5e-4,
        sigma_bg: float = 5e-5,
        sigma_v: float = 0.02,
    ) -> None:
        self.s = state
        self._Qc = self._build_Qc(sigma_a, sigma_g, sigma_ba, sigma_bg)
        self._R_vel = np.eye(3) * sigma_v ** 2

    @staticmethod
    def _build_Qc(sa: float, sg: float, sba: float, sbg: float) -> np.ndarray:
        """Continuous process-noise covariance (12×12 IMU noise space)."""
        return np.diag([sa]*3 + [sg]*3 + [sba]*3 + [sbg]*3) ** 2

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        s = self.s
        R = quat_to_rot(s.q)                    # body → world
        a_c = accel - s.ba                       # bias-corrected accel
        w_c = gyro - s.bg                        # bias-corrected gyro

        # --- Propagate nominal state ---
        v_new = s.v + (R @ a_c + GRAVITY) * dt
        p_new = s.p + s.v * dt + 0.5 * (R @ a_c + GRAVITY) * dt**2

        # Attitude: q_new = q ⊗ Δq(ω_c, dt)
        angle = np.linalg.norm(w_c) * dt
        dq = quat_from_axis_angle(w_c, angle)
        q_new = quat_mult(s.q, dq)
        q_new /= np.linalg.norm(q_new)

        s.p, s.v, s.q = p_new, v_new, q_new

        # --- Error-state transition matrix F (15×15) ---
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)                  # δp_dot = δv
        F[3:6, 6:9] = -R @ skew(a_c)             # δv_dot / δφ
        F[3:6, 9:12] = -R                         # δv_dot / δba
        F[6:9, 6:9] = -skew(w_c)                  # δφ_dot / δφ
        F[6:9, 12:15] = -np.eye(3)                # δφ_dot / δbg

        # First-order discrete: Φ = I + F*dt
        Phi = np.eye(15) + F * dt

        # --- Process noise: Q = G Qc G^T * dt ---
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R                          # accel noise → velocity
        G[6:9, 3:6] = -np.eye(3)                  # gyro noise → attitude
        G[9:12, 6:9] = np.eye(3)                  # accel bias noise
        G[12:15, 9:12] = np.eye(3)                # gyro bias noise

        Q_d = G @ self._Qc @ G.T * dt

        s.P = Phi @ s.P @ Phi.T + Q_d

    def update_velocity(self, vel_meas: np.ndarray, R_meas: np.ndarray | None = None) -> None:
        """Correct EKF using a velocity measurement in world frame."""
        s = self.s
        R_meas = R_meas if R_meas is not None else self._R_vel

        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)                   # measurement selects velocity error

        S = H @ s.P @ H.T + R_meas
        K = s.P @ H.T @ np.linalg.inv(S)

        innov = vel_meas - s.v
        dx = K @ innov

        # --- Inject error into nominal state ---
        s.p += dx[0:3]
        s.v += dx[3:6]
        dq = quat_from_rotvec(dx[6:9])
        s.q = quat_mult(dq, s.q)
        s.q /= np.linalg.norm(s.q)
        s.ba += dx[9:12]
        s.bg += dx[12:15]

        # --- Update covariance (Joseph form for numerical stability) ---
        I_KH = np.eye(15) - K @ H
        s.P = I_KH @ s.P @ I_KH.T + K @ R_meas @ K.T

    def update_delta_v(
        self,
        v_start: np.ndarray,
        delta_v_pred: np.ndarray,
        sigma_dv: float = 0.3,
    ) -> None:
        """
        Update EKF using TCN-predicted delta_v.
        Measurement: v_current ≈ v_start + delta_v_pred
        """
        vel_meas = v_start + delta_v_pred
        R_meas = np.eye(3) * sigma_dv ** 2
        self.update_velocity(vel_meas, R_meas)


def init_from_static(
    accel_static: np.ndarray,
    vel_init: np.ndarray,
    sigma_a: float = 0.05,
    sigma_g: float = 0.005,
    sigma_ba: float = 5e-4,
    sigma_bg: float = 5e-5,
    sigma_v: float = 0.02,
) -> EKF15:
    """
    Build an EKF initialized from a static accelerometer reading.
    Velocity is initialized from ground-truth (first Leica sample).
    """
    q0 = quat_from_gravity(accel_static)

    # Initial covariance — generous; filter will converge quickly
    P0 = np.diag([
        1.0, 1.0, 1.0,           # position uncertainty (m²)
        0.1, 0.1, 0.1,           # velocity (m/s)²
        (0.1)**2, (0.1)**2, (0.5)**2,  # attitude (rad²): roll/pitch tighter than yaw
        (0.5)**2, (0.5)**2, (0.5)**2,  # accel bias (m/s²)²
        (0.05)**2, (0.05)**2, (0.05)**2,  # gyro bias (rad/s)²
    ])

    state = EKFState(
        p=np.zeros(3),
        v=vel_init.copy(),
        q=q0,
        ba=np.zeros(3),
        bg=np.zeros(3),
        P=P0,
    )
    return EKF15(state, sigma_a=sigma_a, sigma_g=sigma_g,
                 sigma_ba=sigma_ba, sigma_bg=sigma_bg, sigma_v=sigma_v)
