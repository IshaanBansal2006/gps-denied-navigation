"""Navigation eval with RLS-adapted v15 head.

Pipeline:
  1. Load v15 LSTM + per-channel norm stats.
  2. Initialize an RLSHead from the pre-trained linear head weights.
  3. Walk the pre-outage portion of the test sequence: at each STRIDE-th sample
     extract the LSTM hidden state, get the GPS-aided velocity target, and
     `update(x, y)` the RLS head. This specializes the head to the sequence
     while the LSTM body stays frozen.
  4. At outage onset, predict velocity using the *adapted* head (no further
     updates during the outage — GPS is unavailable).
  5. Wrap predictions in the same velocity-only filter that v15 uses to land
     a fair comparison.

Outputs:
  results/lstm_v15_rls/test_metrics.json — final + mean position/velocity errors
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.adaptation.rls import RLSHead
from src.filters.ekf import init_from_static

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RESULTS_DIR = PROJECT_ROOT / "results" / "lstm_v15_rls"

STRIDE = 25
SIGMA_PROCESS = 0.5
R_VELONLY = np.diag([0.61 ** 2, 0.67 ** 2, 0.25 ** 2])


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 output_dim=3, dropout=0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.head(self.dropout(out)), state


def load_lstm_v15(device):
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "lstm_v15.pt",
                      map_location=device, weights_only=False)
    c = ckpt.get("config", {})
    model = LSTMRegressor(
        input_size=c.get("input_size", 6),
        hidden_size=c.get("hidden_size", 128),
        num_layers=c.get("num_layers", 2),
        output_dim=c.get("output_dim", 3),
        dropout=c.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ns = ckpt["norm_stats"]
    norm = {
        "x_mean": np.array(ns["x_mean"], dtype=np.float32),
        "x_std":  np.array(ns["x_std"],  dtype=np.float32),
        "y_mean": np.array(ns["y_mean"], dtype=np.float32),
        "y_std":  np.array(ns["y_std"],  dtype=np.float32),
    }
    return model, norm


@torch.no_grad()
def lstm_step_features(model: LSTMRegressor, imu_sample: np.ndarray, norm: dict,
                        device: torch.device, state):
    """Run one LSTM step. Return (hidden_features, head_velocity, new_state).

    hidden_features: 1D numpy array of size hidden_size (post-dropout-disabled),
    head_velocity: 3D numpy array (un-normalized, m/s).
    """
    x = (imu_sample.astype(np.float32) - norm["x_mean"]) / norm["x_std"]
    t = torch.tensor(x[None, None, :], dtype=torch.float32, device=device)
    out, new_state = model.lstm(t, state)
    h = out[0, 0, :]  # hidden_size
    y_norm = model.head(h)
    v = y_norm.cpu().numpy() * norm["y_std"] + norm["y_mean"]
    return h.cpu().numpy(), v, new_state


def warm_lstm_with_rls(model, imu_data, gt_vel, norm, device,
                       rls: RLSHead, end_idx: int):
    """Walk pre-outage data, updating RLS at every STRIDE-th sample."""
    state = None
    last_v = np.zeros(3, dtype=np.float32)
    last_h = None
    steps = 0
    for k in range(end_idx):
        h, v, state = lstm_step_features(model, imu_data[k], norm, device, state)
        last_h, last_v = h, v
        steps += 1
        if steps >= STRIDE:
            steps = 0
            # Update RLS with the GPS-aided velocity at this step.
            y_target = (gt_vel[k].astype(np.float32) - norm["y_mean"]) / norm["y_std"]
            rls.update(h, y_target)
    return last_v, state, last_h


def run_velonly_filter_with_rls(model, rls, imu_data, timestamps, norm, device,
                                 vel_init, start, end, lstm_state0):
    """Same velocity-only filter as v15, but use the *adapted* head for predictions."""
    v = vel_init.copy().astype(np.float64)
    P = np.eye(3) * 0.5
    Q_rate = SIGMA_PROCESS ** 2 * np.eye(3)

    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = v
    state = lstm_state0
    steps = 0
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        P = P + Q_rate * dt
        h, _v_pretrained, state = lstm_step_features(model, imu_data[k], norm, device, state)
        v_pred_norm = rls.predict(h)
        v_pred = v_pred_norm * norm["y_std"] + norm["y_mean"]
        steps += 1
        if steps >= STRIDE:
            steps = 0
            S = P + R_VELONLY
            K = P @ np.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (np.eye(3) - K) @ P
        vel[i + 1] = v
    return vel


def evaluate_outage(df, model, norm, device, outage_s: float, forgetting: float, p_init: float):
    timestamps = df["timestamp"].to_numpy()
    accel  = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro   = df[["gyro_x",  "gyro_y",  "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x", "gt_vel_y", "gt_vel_z"]].to_numpy()
    imu_data = df[["gyro_x", "gyro_y", "gyro_z",
                   "accel_x", "accel_y", "accel_z"]].to_numpy()

    duration = timestamps[-1] - timestamps[0]
    outage_start_t = duration * 0.4
    outage_start = int(np.searchsorted(timestamps - timestamps[0], outage_start_t))
    outage_end   = min(len(df) - 1,
                       int(np.searchsorted(timestamps - timestamps[0], outage_start_t + outage_s)))

    # Initialize RLS from the v15 head weights.
    head_W = model.head.weight.detach().cpu().numpy().T  # (in_dim, out_dim)
    head_b = model.head.bias.detach().cpu().numpy()
    rls = RLSHead(in_dim=head_W.shape[0], out_dim=head_W.shape[1],
                  forgetting=forgetting, p_init=p_init)
    rls.reset(head_W, head_b)

    print(f"  Warming LSTM + RLS over {outage_start} pre-outage samples...", end=" ", flush=True)
    _, lstm_state0, _ = warm_lstm_with_rls(model, imu_data, gt_vel, norm, device,
                                            rls, outage_start)
    print(f"done ({rls.n_updates} RLS updates)")

    vel_at_outage = gt_vel[outage_start]
    gt_outage = gt_vel[outage_start:outage_end + 1]

    def ferr(v):
        n = min(len(v), len(gt_outage)) - 1
        return float(np.linalg.norm(v[n] - gt_outage[n]))

    def merr(v):
        n = min(len(v), len(gt_outage))
        return float(np.mean(np.linalg.norm(v[:n] - gt_outage[:n], axis=1)))

    v_rls = run_velonly_filter_with_rls(model, rls, imu_data, timestamps, norm, device,
                                         vel_at_outage, outage_start, outage_end, lstm_state0)

    return {
        "outage_s": outage_s,
        "rls_final": ferr(v_rls),
        "rls_mean":  merr(v_rls),
        "rls_n_updates": rls.n_updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60",
                        help="comma-separated outage durations in seconds")
    parser.add_argument("--forgetting", type=float, default=0.995,
                        help="RLS forgetting factor (default 0.995)")
    parser.add_argument("--p-init", type=float, default=0.1,
                        help="RLS initial covariance diagonal — controls how aggressively "
                             "the head adapts. 0.1 sweet-spot for MH_05 30s (decision 029).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, norm = load_lstm_v15(device)
    print(f"Loaded lstm_v15.pt: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"RLS forgetting factor: {args.forgetting}\n")

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"Sequence: {args.seq}  |  {len(df)} samples\n")

    outage_durations = [float(x) for x in args.outages.split(",")]
    all_results = []
    for t_out in outage_durations:
        print(f"--- Outage {t_out}s ---")
        r = evaluate_outage(df, model, norm, device, t_out, args.forgetting, args.p_init)
        print(f"  RLS-adapted v15+filter:  final={r['rls_final']:.4f}  mean={r['rls_mean']:.4f}")
        all_results.append(r)

    out = {
        "sequence": args.seq,
        "forgetting": args.forgetting,
        "p_init": args.p_init,
        "by_outage": all_results,
    }
    out_path = RESULTS_DIR / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
