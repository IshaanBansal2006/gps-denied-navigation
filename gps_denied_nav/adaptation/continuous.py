"""Continuous adaptation during a GPS outage.

RLS (decision 029) freezes the moment GPS disappears. This module keeps the
linear head updating *during* the outage using self-supervised signals.

Two pseudo-targets, blended:

1. **Smoothed filter velocity** (self-distillation). The velocity-only Kalman
   filter naturally smooths predictions over time. Pulling the head toward
   the filter's smoothed output is a regularizer that biases predictions
   toward consistency with the rolling average.

2. **Gyro-rotated previous velocity** (physics consistency). If we assume
   negligible body-frame acceleration over one STRIDE step, then
       v_world_t ≈ R(omega * dt) @ v_world_{t-1}
   The gyro gives us omega directly; the rotation tells us where the
   velocity vector *should* point if motion is purely rotational. Mismatch
   between predicted-velocity-rotated-by-gyro and current prediction is
   evidence of model error in a direction we can correct.

The pseudo-target is a convex combination::

    target = alpha_smooth * v_filter_ema  +  (1 - alpha_smooth) * v_gyro_rotated

Hyperparameters (selected on val MH_04, decision 032):

    alpha_smooth = 0.7
    ema_alpha    = 0.95
    outage_lambda = 0.999  (much larger than pre-outage RLS to prevent runaway
                            adaptation off self-supervised pseudo-targets)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rls import RLSHead


def _rodrigues(omega_dt: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation formula. omega_dt is a 3-vector (rad)."""
    theta = float(np.linalg.norm(omega_dt))
    if theta < 1e-8:
        return np.eye(3)
    k = omega_dt / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


@dataclass
class ContinuousAdapter:
    """Wrap an RLSHead with self-supervised updates during the GPS outage.

    During pre-outage, the inner ``rls`` head is updated normally (with GPS-
    aided ground truth) via the standard NavPipeline.warmup path. At outage
    onset, callers should call ``reset()`` to seed the EMA. During the
    outage, callers should call ``update_during_outage()`` at every
    STRIDE-th sample with the current LSTM hidden state, the current
    filter velocity estimate (un-normalized), and the gyro reading.

    Attributes
    ----------
    rls:
        The inner adaptive head.
    alpha_smooth:
        Convex weight on the smoothed-filter pseudo-target vs the
        gyro-rotated pseudo-target. 1.0 = pure smoothness; 0.0 = pure gyro.
    ema_alpha:
        EMA weight on the smoothed filter velocity. Closer to 1.0 = more
        history weight, slower response.
    outage_lambda:
        Forgetting factor override used during outage updates. Should be
        ~0.999 — much closer to 1.0 than pre-outage's 0.995 — because we
        don't trust the self-supervised target as much as GPS.
    """

    rls: RLSHead
    alpha_smooth: float = 0.7
    ema_alpha: float = 0.95
    outage_lambda: float = 0.999

    _ema_v_norm: Optional[np.ndarray] = None
    _last_v_world: Optional[np.ndarray] = None
    _saved_lambda: Optional[float] = None

    def reset(
        self,
        v_init_norm: np.ndarray,
        v_init_world: np.ndarray,
    ) -> None:
        """Call at the start of the outage to seed the EMA."""
        self._ema_v_norm = v_init_norm.astype(np.float64).copy()
        self._last_v_world = v_init_world.astype(np.float64).copy()
        # Soften forgetting during outage — pseudo-targets are less reliable.
        self._saved_lambda = self.rls.lam
        self.rls.lam = float(self.outage_lambda)

    def restore(self) -> None:
        if self._saved_lambda is not None:
            self.rls.lam = self._saved_lambda
            self._saved_lambda = None

    def update_during_outage(
        self,
        h: np.ndarray,
        v_filter_world: np.ndarray,
        gyro: np.ndarray,
        dt: float,
        norm: dict,
    ) -> None:
        """One self-supervised update step.

        Parameters
        ----------
        h:           LSTM hidden features for this step (in_dim,).
        v_filter_world: Current filter velocity estimate in world frame (m/s).
        gyro:        Body-frame angular velocity for this step (rad/s).
        dt:          Time step (s) since last update.
        norm:        Normalization stats dict (uses y_mean, y_std).
        """
        # 1) Smoothed-filter pseudo-target (normalized space).
        v_filter_norm = ((v_filter_world.astype(np.float64) - norm["y_mean"])
                         / norm["y_std"])
        self._ema_v_norm = (self.ema_alpha * self._ema_v_norm
                            + (1.0 - self.ema_alpha) * v_filter_norm)

        # 2) Gyro-rotated previous-velocity pseudo-target (world → normalized).
        # Note: gyro is in body frame; rotating world-frame velocity by
        # R(-gyro*dt) approximates how the velocity vector should evolve if
        # motion were purely rotational.
        R = _rodrigues(-gyro.astype(np.float64) * dt)
        v_gyro_world = R @ self._last_v_world
        v_gyro_norm = (v_gyro_world - norm["y_mean"]) / norm["y_std"]

        # 3) Blend.
        target_norm = (self.alpha_smooth * self._ema_v_norm
                       + (1.0 - self.alpha_smooth) * v_gyro_norm)

        # 4) RLS update with the pseudo-target.
        self.rls.update(h.astype(np.float64), target_norm.astype(np.float32))

        # 5) Roll the velocity for next gyro-prediction step.
        self._last_v_world = v_filter_world.astype(np.float64).copy()

    def predict(self, h: np.ndarray) -> np.ndarray:
        return self.rls.predict(h)
