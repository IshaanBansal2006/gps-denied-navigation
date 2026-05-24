"""NavPipeline — composable model + adapter + filter for GPS-denied navigation.

Usage example::

    from gps_denied_nav import NavPipeline, EuRoCSequence
    from gps_denied_nav.models import LSTMRegressor, load_lstm_checkpoint
    from gps_denied_nav.adaptation import RLSHead
    from gps_denied_nav.filters import VelocityOnlyFilter
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = EuRoCSequence.load("MH_05_difficult", "data/sequences")
    model, norm = load_lstm_checkpoint("checkpoints/lstm_v15.pt", device)
    adapter = RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
    pipeline = NavPipeline(model=model, adapter=adapter,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25)

    outage_start, outage_end = seq.outage_window(start_frac=0.6, duration_s=30.0)
    result = pipeline.run_outage(seq, outage_start, outage_end)
    print(f"Final velocity error: {result.final_error_norm:.3f} m/s")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .adaptation.continuous import ContinuousAdapter
from .adaptation.rls import RLSHead
from .data.euroc import EuRoCSequence
from .filters.velocity_only import VelocityOnlyFilter
from .models.lstm import LSTMRegressor


@dataclass
class OutageResult:
    """Per-step trajectory + metrics for one outage run."""
    velocity_estimate: np.ndarray  # (T+1, 3) — filtered output during outage
    velocity_gt: np.ndarray        # (T+1, 3) — ground truth for comparison
    velocity_lstm_raw: np.ndarray  # (T, 3)   — model predictions before filter
    position_estimate: np.ndarray  # (T+1, 3) — integrated estimate
    position_gt: np.ndarray        # (T+1, 3) — integrated truth
    timestamps: np.ndarray         # (T+1,)

    @property
    def velocity_error(self) -> np.ndarray:
        """Per-step velocity error magnitude (m/s)."""
        n = min(len(self.velocity_estimate), len(self.velocity_gt))
        return np.linalg.norm(
            self.velocity_estimate[:n] - self.velocity_gt[:n], axis=1)

    @property
    def position_error(self) -> np.ndarray:
        """Per-step position drift magnitude (m)."""
        n = min(len(self.position_estimate), len(self.position_gt))
        return np.linalg.norm(
            self.position_estimate[:n] - self.position_gt[:n], axis=1)

    @property
    def final_error_norm(self) -> float:
        """Final velocity error at outage end (m/s)."""
        return float(self.velocity_error[-1])

    @property
    def mean_error_norm(self) -> float:
        """Mean velocity error across the outage rollout (m/s)."""
        return float(self.velocity_error.mean())

    @property
    def final_position_drift(self) -> float:
        """Final position drift at outage end (m)."""
        return float(self.position_error[-1])


class NavPipeline:
    """Composable inference pipeline: model + (optional) adapter + filter.

    The pipeline runs an LSTM body over IMU samples, optionally adapts the
    final linear head from pre-outage GPS-aided velocity targets (via the
    adapter), then feeds head predictions through a velocity-only filter
    during the outage. Output is a velocity time-series suitable for
    downstream integration into position.

    Parameters
    ----------
    model:
        Pre-trained LSTM regressor whose `.head` will be the adaptation
        target if an adapter is provided.
    adapter:
        Optional ``RLSHead`` (or subclass) that adapts the final linear
        head using pre-outage data. ``None`` = use the pre-trained head
        as-is.
    filter:
        Velocity-only Kalman filter that smooths the head predictions
        during the outage.
    norm:
        Train-set normalization stats: ``x_mean, x_std, y_mean, y_std``.
    device:
        Torch device for model inference.
    update_stride:
        Filter update / adapter update cadence in IMU samples (default 25
        = 8 Hz at 200 Hz IMU rate).
    """

    def __init__(
        self,
        model: LSTMRegressor,
        adapter: Optional[RLSHead],
        filter: VelocityOnlyFilter,
        norm: dict,
        device: torch.device,
        update_stride: int = 25,
        continuous_adapter: Optional[ContinuousAdapter] = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.filter = filter
        self.norm = norm
        self.device = device
        self.update_stride = update_stride
        self.continuous_adapter = continuous_adapter

        if adapter is not None:
            head_W = model.head.weight.detach().cpu().numpy().T
            head_b = model.head.bias.detach().cpu().numpy()
            adapter.reset(head_W, head_b)
        if continuous_adapter is not None:
            assert adapter is continuous_adapter.rls, \
                "continuous_adapter.rls must reference the same RLSHead passed as `adapter`"

    @torch.no_grad()
    def _step_features(
        self,
        imu_sample: np.ndarray,
        state,
    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """One LSTM step → (hidden_features, head_velocity, new_state)."""
        x = (imu_sample.astype(np.float32) - self.norm["x_mean"]) / self.norm["x_std"]
        t = torch.tensor(x[None, None, :], dtype=torch.float32, device=self.device)
        out, new_state = self.model.lstm(t, state)
        h = out[0, 0, :]
        y_norm = self.model.head(h)
        v = y_norm.cpu().numpy() * self.norm["y_std"] + self.norm["y_mean"]
        return h.cpu().numpy(), v, new_state

    @torch.no_grad()
    def warmup(
        self,
        sequence: EuRoCSequence,
        end_idx: int,
    ) -> Optional[tuple]:
        """Run LSTM over samples [0, end_idx) to warm hidden state.

        If an adapter is configured, also updates it from ground-truth velocity
        at every ``update_stride``-th sample.
        Returns the final LSTM state (h, c).
        """
        state = None
        steps = 0
        for k in range(end_idx):
            h, _v, state = self._step_features(sequence.imu[k], state)
            steps += 1
            if self.adapter is not None and steps >= self.update_stride:
                steps = 0
                y_target = ((sequence.gt_vel[k].astype(np.float32) - self.norm["y_mean"])
                            / self.norm["y_std"])
                self.adapter.update(h, y_target)
        return state

    def _predict_velocity(self, hidden: np.ndarray) -> np.ndarray:
        """Use the adapted head if available, otherwise the original."""
        if self.adapter is None:
            with torch.no_grad():
                t = torch.tensor(hidden, device=self.device)
                y_norm = self.model.head(t).cpu().numpy()
            return y_norm * self.norm["y_std"] + self.norm["y_mean"]
        v_norm = self.adapter.predict(hidden)
        return v_norm * self.norm["y_std"] + self.norm["y_mean"]

    def run_outage(
        self,
        sequence: EuRoCSequence,
        outage_start: int,
        outage_end: int,
        lstm_state_at_outage: Optional[tuple] = None,
    ) -> OutageResult:
        """End-to-end inference over an outage window.

        If ``lstm_state_at_outage`` is None, runs ``warmup`` first.
        Returns an ``OutageResult`` with full trajectories.
        """
        if lstm_state_at_outage is None:
            lstm_state_at_outage = self.warmup(sequence, outage_start)

        v_init = sequence.gt_vel[outage_start]
        self.filter.reset(v_init)
        if self.continuous_adapter is not None:
            v_init_norm = ((v_init.astype(np.float64) - self.norm["y_mean"])
                           / self.norm["y_std"]).astype(np.float64)
            self.continuous_adapter.reset(v_init_norm, v_init)

        timestamps = sequence.timestamps[outage_start:outage_end + 1]
        gt_vel = sequence.gt_vel[outage_start:outage_end + 1]
        n_outage = outage_end - outage_start

        vel_est = np.zeros((n_outage + 1, 3), dtype=np.float64)
        vel_lstm_raw = np.zeros((n_outage, 3), dtype=np.float64)
        vel_est[0] = self.filter.velocity

        state = lstm_state_at_outage
        steps = 0
        for i in range(n_outage):
            k = outage_start + i
            dt = float(sequence.timestamps[k + 1] - sequence.timestamps[k])
            if dt <= 0 or dt > 0.05:
                dt = 0.005
            self.filter.predict(dt)
            h, _v_pretrained, state = self._step_features(sequence.imu[k], state)
            v_pred = self._predict_velocity(h)
            vel_lstm_raw[i] = v_pred

            steps += 1
            if steps >= self.update_stride:
                steps = 0
                self.filter.update(v_pred)
                # Self-supervised continuous adaptation kicks in here.
                if self.continuous_adapter is not None:
                    gyro_k = sequence.imu[k, :3]
                    self.continuous_adapter.update_during_outage(
                        h, self.filter.velocity, gyro_k,
                        dt * self.update_stride, self.norm,
                    )

            vel_est[i + 1] = self.filter.velocity

        if self.continuous_adapter is not None:
            self.continuous_adapter.restore()

        # Trapezoidal integration to position (origin at outage start).
        dt_arr = np.diff(timestamps)
        dt_arr = np.where((dt_arr > 0) & (dt_arr < 0.05), dt_arr, 0.005)
        pos_est = np.zeros((n_outage + 1, 3))
        pos_gt = np.zeros((n_outage + 1, 3))
        for i in range(n_outage):
            pos_est[i + 1] = pos_est[i] + 0.5 * (vel_est[i] + vel_est[i + 1]) * dt_arr[i]
            pos_gt[i + 1] = pos_gt[i] + 0.5 * (gt_vel[i] + gt_vel[i + 1]) * dt_arr[i]

        return OutageResult(
            velocity_estimate=vel_est,
            velocity_gt=gt_vel.astype(np.float64),
            velocity_lstm_raw=vel_lstm_raw,
            position_estimate=pos_est,
            position_gt=pos_gt,
            timestamps=timestamps,
        )
