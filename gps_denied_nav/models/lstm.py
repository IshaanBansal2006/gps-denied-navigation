"""LSTM regressor for IMU → velocity prediction.

This is the shared architecture used by v12, v13, v14, v15, v16, v17 (and v18
with larger defaults). Sequence-style: feed one IMU sample, carry hidden
state forward at 200 Hz, emit a velocity prediction at the head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM with a linear regression head.

    Parameters
    ----------
    input_size:
        Number of IMU channels (default 6 = gyro xyz + accel xyz).
    hidden_size:
        Per-layer LSTM hidden state size. v12-v17 use 128; v18 uses 256.
    num_layers:
        Stacked LSTM depth. v12-v17 use 2; v18 uses 3.
    output_dim:
        Output dimensionality (default 3 = velocity xyz).
    dropout:
        Inter-layer dropout (only effective when num_layers > 1) and applied
        to the LSTM output before the final head.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, state = self.lstm(x, state)
        return self.head(self.dropout(out)), state


def load_lstm_checkpoint(
    path: Path | str,
    device: torch.device,
) -> Tuple[LSTMRegressor, dict]:
    """Load a checkpoint produced by any of the train_lstm_v* scripts.

    Returns (model, norm_stats_dict) where norm_stats has float32 numpy arrays
    keyed ``x_mean, x_std, y_mean, y_std``.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = LSTMRegressor(
        input_size=cfg.get("input_size", 6),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        output_dim=cfg.get("output_dim", 3),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ns_raw = ckpt["norm_stats"]
    norm = {
        "x_mean": np.array(ns_raw["x_mean"], dtype=np.float32),
        "x_std":  np.array(ns_raw["x_std"],  dtype=np.float32),
        "y_mean": np.array(ns_raw["y_mean"], dtype=np.float32),
        "y_std":  np.array(ns_raw["y_std"],  dtype=np.float32),
    }
    return model, norm
