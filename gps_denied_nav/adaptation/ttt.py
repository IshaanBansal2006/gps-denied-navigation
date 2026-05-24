"""Test-Time Training adapter.

Right before a GPS outage begins, run K gradient steps over the LSTM body
using the most recent (IMU, GPS-aided-velocity) pairs. Unlike RLS, which
only updates the final linear head, TTT can rewrite the full feature
extractor. The trade-off is catastrophic-forgetting risk — too many steps
or too high a learning rate and the model overfits the warmup window.

After the outage, the original weights must be restored before evaluating on
another sequence. Use `TTTAdapter` as a context manager or call
`restore()` explicitly.

Typical hyperparameters (selected on val MH_04, decision 031):
  K=10, lr=5e-5, freeze_lstm_layers=0 (i.e. update everything)

Usage::

    ttt = TTTAdapter(model, lr=5e-5, K=10)
    with ttt.adapt(sequence_imu, sequence_gt_vel, end_idx=outage_start, norm=norm):
        # model is now specialized — run nav eval here
        result = pipeline.run_outage(...)
    # weights restored on context exit
"""
from __future__ import annotations

import contextlib
import copy
from typing import Optional

import numpy as np
import torch
from torch import nn


WINDOW_LEN = 200   # one-second IMU windows used as TTT training samples


class TTTAdapter:
    """Test-time training adapter for a pre-trained LSTM regressor.

    Parameters
    ----------
    model:
        The pre-trained LSTM model. Will be modified in-place during adapt().
    K:
        Number of gradient steps over the warmup window. Larger = more
        specialization, larger forgetting risk.
    lr:
        Learning rate for the inner-loop optimizer.
    freeze_lstm_layers:
        Number of LSTM layers to freeze (0 = all unfrozen). Freezing the
        first layer keeps low-level IMU features intact while adapting higher
        layers.
    window_len:
        Length of each pre-outage sample window. Default 200 samples (1s at
        200 Hz) mirrors the training-window convention.
    """

    def __init__(
        self,
        model: nn.Module,
        K: int = 10,
        lr: float = 5e-5,
        freeze_lstm_layers: int = 0,
        window_len: int = WINDOW_LEN,
    ) -> None:
        self.model = model
        self.K = int(K)
        self.lr = float(lr)
        self.freeze_lstm_layers = int(freeze_lstm_layers)
        self.window_len = int(window_len)
        self._snapshot: Optional[dict] = None

    def _snapshot_state(self) -> None:
        self._snapshot = copy.deepcopy(self.model.state_dict())

    def restore(self) -> None:
        if self._snapshot is None:
            return
        self.model.load_state_dict(self._snapshot)
        self._snapshot = None

    def _set_trainable(self) -> None:
        # Freeze the first `freeze_lstm_layers` LSTM weight tensors.
        # nn.LSTM has 4 weight tensors per layer: weight_ih_l<i>, weight_hh_l<i>,
        # bias_ih_l<i>, bias_hh_l<i>. We freeze by index.
        for name, p in self.model.named_parameters():
            p.requires_grad = True
        for i in range(self.freeze_lstm_layers):
            for k in (f"weight_ih_l{i}", f"weight_hh_l{i}",
                      f"bias_ih_l{i}", f"bias_hh_l{i}"):
                if hasattr(self.model.lstm, k):
                    getattr(self.model.lstm, k).requires_grad = False

    def _build_windows(
        self,
        imu_norm: np.ndarray,
        vel_norm: np.ndarray,
        end_idx: int,
        device: torch.device,
        n_windows: int = 32,
    ):
        """Sample n_windows random pre-outage IMU windows for TTT."""
        max_start = end_idx - self.window_len
        if max_start <= 0:
            raise ValueError("end_idx too small for TTT window sampling")
        rng = np.random.default_rng(42)
        starts = rng.integers(0, max_start, size=n_windows)
        xs = np.stack([imu_norm[s:s + self.window_len] for s in starts])
        ys = np.stack([vel_norm[s + self.window_len - 1] for s in starts])
        return (torch.tensor(xs, dtype=torch.float32, device=device),
                torch.tensor(ys, dtype=torch.float32, device=device))

    @contextlib.contextmanager
    def adapt(
        self,
        imu: np.ndarray,
        gt_vel: np.ndarray,
        end_idx: int,
        norm: dict,
        device: torch.device,
        n_windows: int = 32,
        verbose: bool = False,
    ):
        """Specialize the model to the most-recent pre-outage data.

        Yields control to the caller with the model adapted; restores the
        original weights on exit.
        """
        self._snapshot_state()
        self._set_trainable()

        imu_norm = ((imu.astype(np.float32) - norm["x_mean"]) / norm["x_std"])
        vel_norm = ((gt_vel.astype(np.float32) - norm["y_mean"]) / norm["y_std"])

        xs, ys = self._build_windows(imu_norm, vel_norm, end_idx, device, n_windows)
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        self.model.train()
        initial_loss = None
        for step in range(self.K):
            pred, _ = self.model(xs)
            # Per-window: last-timestep prediction vs end-of-window GT.
            last_pred = pred[:, -1, :]
            loss = ((last_pred - ys) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if initial_loss is None:
                initial_loss = float(loss.item())
            if verbose:
                print(f"  TTT step {step + 1}/{self.K}: loss={float(loss.item()):.4f}")

        self.model.eval()
        try:
            yield {
                "n_steps": self.K,
                "lr": self.lr,
                "n_windows": n_windows,
                "initial_loss": initial_loss,
                "final_loss": float(loss.item()) if self.K > 0 else None,
            }
        finally:
            self.restore()
