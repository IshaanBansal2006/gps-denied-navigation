"""Recursive Least Squares adaptive linear head.

Drop-in replacement for the final ``nn.Linear`` of an LSTM regressor that can
be specialized to a specific sequence using pre-outage GPS-aided velocity
observations, then used to predict during a GPS outage with the adapted
weights.

Math (multi-output linear regression, streaming update):
  y_hat = W^T x + b
  e_k  = y_k - (W^T x_k + b)
  K_k  = (P_{k-1} x_k) / (lambda + x_k^T P_{k-1} x_k)
  W_k  = W_{k-1} + K_k e_k^T
  P_k  = (P_{k-1} - K_k x_k^T P_{k-1}) / lambda

`lambda` is the forgetting factor (1.0 = no forgetting, recommended 0.99-0.999).
The bias `b` is updated by the running mean of `y - W^T x` over the warmup window.

All operations are pure NumPy (no autograd) so this is cheap at inference time.
"""
from __future__ import annotations

import numpy as np


class RLSHead:
    """Online-adaptive linear head with forgetting factor.

    Parameters
    ----------
    in_dim : int
        LSTM hidden-state dimensionality (e.g., 128).
    out_dim : int
        Target dimensionality (e.g., 3 for x/y/z velocity).
    forgetting : float
        Forgetting factor lambda in (0, 1]. 1.0 = pure RLS. Smaller values
        weight recent samples more heavily; for our 10-second pre-outage
        warmup (~2000 samples at 200 Hz, ~80 RLS updates at STRIDE=25),
        0.995 was found to give the best balance between specialization and
        retention of the pre-trained head.
    p_init : float
        Initial covariance diagonal (large value => weak prior, strong learning).
    """

    def __init__(self, in_dim: int, out_dim: int,
                 forgetting: float = 0.995, p_init: float = 100.0) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lam = float(forgetting)
        self.p_init = float(p_init)
        self.W = np.zeros((in_dim, out_dim), dtype=np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        self.P = np.eye(in_dim, dtype=np.float64) * self.p_init
        self._n_updates = 0

    def reset(self, W0: np.ndarray, b0: np.ndarray) -> None:
        """Re-initialize from a pre-trained linear head.

        W0 has shape (in_dim, out_dim) following torch's (in_dim, out_dim)
        memory layout for nn.Linear's `weight.T` — i.e., y = x @ W0 + b0.
        """
        assert W0.shape == (self.in_dim, self.out_dim), \
            f"W0 shape {W0.shape} != ({self.in_dim}, {self.out_dim})"
        assert b0.shape == (self.out_dim,)
        self.W = W0.astype(np.float64).copy()
        self.b = b0.astype(np.float64).copy()
        self.P = np.eye(self.in_dim, dtype=np.float64) * self.p_init
        self._n_updates = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Ingest one (feature, target) pair from pre-outage data."""
        assert x.shape == (self.in_dim,), f"x shape {x.shape} != ({self.in_dim},)"
        assert y.shape == (self.out_dim,)

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        K = Px / denom

        # error = y - (x @ W + b)
        y_hat = self.W.T @ x + self.b
        e = y - y_hat

        self.W = self.W + np.outer(K, e)
        self.P = (self.P - np.outer(K, Px)) / self.lam

        # Track running mean residual for bias update.
        # Decayed exponential mean with the same forgetting factor.
        self.b = self.lam * self.b + (1.0 - self.lam) * (y - self.W.T @ x)
        self._n_updates += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Produce a target for a single feature (no gradient)."""
        assert x.shape == (self.in_dim,)
        return (self.W.T @ x.astype(np.float64) + self.b).astype(np.float32)

    @property
    def n_updates(self) -> int:
        return self._n_updates
