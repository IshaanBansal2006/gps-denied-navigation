"""Velocity-only Kalman filter.

Three-state filter (just velocity, no position or attitude) that fuses noisy
neural-network velocity predictions into a smoother estimate during a GPS
outage. Decision 019 established this beats the full 15-state strapdown EKF
during GPS-denied flight because attitude drift poisons IMU propagation
within 10 s.

Math (constant-velocity dynamics model, R diagonal):
  Predict:  v_t = v_{t-1};  P_t = P_{t-1} + Q * dt
  Update:   K = P (P + R)^{-1};  v += K (z - v);  P = (I - K) P
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_R_DIAG: Sequence[float] = (0.61 ** 2, 0.67 ** 2, 0.25 ** 2)


@dataclass
class VelocityOnlyFilter:
    """Three-state Kalman filter on velocity only.

    Parameters
    ----------
    sigma_process:
        Process-noise standard deviation (m/s per √s). Default 0.5 was
        selected empirically by decision 018.
    R_diag:
        Per-channel measurement noise variances (m/s)². Default reflects the
        per-axis residual std observed during model training.
    P_init:
        Initial covariance diagonal. Larger = less trust in the initial v.
    """

    sigma_process: float = 0.5
    R_diag: Sequence[float] = field(default_factory=lambda: list(DEFAULT_R_DIAG))
    P_init: float = 0.5

    v: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    Q_rate: np.ndarray = field(init=False)
    _I3: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.v = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * self.P_init
        self.R = np.diag(np.asarray(self.R_diag, dtype=np.float64))
        self.Q_rate = (self.sigma_process ** 2) * np.eye(3, dtype=np.float64)
        self._I3 = np.eye(3, dtype=np.float64)

    def reset(self, v_init: np.ndarray) -> None:
        """Re-initialize the filter at the start of an outage."""
        assert v_init.shape == (3,)
        self.v = v_init.astype(np.float64).copy()
        self.P = np.eye(3, dtype=np.float64) * self.P_init

    def predict(self, dt: float) -> None:
        """Propagate covariance forward. (Constant-velocity model.)"""
        self.P = self.P + self.Q_rate * dt

    def update(self, v_meas: np.ndarray) -> None:
        """Fuse a velocity measurement (from the neural network)."""
        assert v_meas.shape == (3,)
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.v = self.v + K @ (v_meas.astype(np.float64) - self.v)
        self.P = (self._I3 - K) @ self.P

    @property
    def velocity(self) -> np.ndarray:
        return self.v.copy()
