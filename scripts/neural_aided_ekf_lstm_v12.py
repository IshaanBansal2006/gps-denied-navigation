"""
Navigation eval using LSTM v12 absolute velocity predictions.

LSTM inference differs from TCN: feed one IMU sample at a time, carry (h, c)
state forward across the entire outage. No cold-start, no window gaps.

TCN update rate: every STRIDE=25 steps.
LSTM update rate: every STRIDE=25 steps (same cadence for fair comparison),
but the LSTM hidden state runs at 200Hz continuously so it always has full context.
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

from src.filters.ekf import EKF15, init_from_static

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RESULTS_DIR = PROJECT_ROOT / "results" / "neural_aided_ekf_lstm_v12"

STRIDE = 25
SIGMA_TCN_SWEEP = [0.01, 0.05, 0.1, 0.3]
SIGMA_PROCESS = 0.5


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


def load_lstm_v12(device):
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "lstm_v12.pt",
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
def lstm_step(model, imu_sample, norm, device, state):
    """Feed one IMU sample, return velocity prediction + new LSTM state."""
    x = (imu_sample.astype(np.float32) - norm["x_mean"]) / norm["x_std"]
    t = torch.tensor(x[None, None, :], dtype=torch.float32, device=device)
    y_norm, new_state = model(t, state)
    v = y_norm[0, 0, :].cpu().numpy() * norm["y_std"] + norm["y_mean"]
    return v, new_state


def warm_lstm(model, imu_data, norm, device, end_idx):
    """Run LSTM over imu_data[0:end_idx] to build up hidden state before outage."""
    state = None
    v_last = None
    for k in range(end_idx):
        v_last, state = lstm_step(model, imu_data[k], norm, device, state)
    return v_last, state


def run_dead_reckoning(accel, timestamps, ba, vel_init, start, end):
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        vel[i + 1] = vel[i] + (accel[k] - ba) * dt
    return vel


def run_standalone_lstm(model, imu_data, norm, device, vel_init, start, end, lstm_state0):
    """LSTM runs at 200Hz; emit velocity every STRIDE steps."""
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = vel_init
    state = lstm_state0
    v_current = vel_init.copy()
    steps = 0
    for i in range(n):
        k = start + i
        v_pred, state = lstm_step(model, imu_data[k], norm, device, state)
        steps += 1
        if steps >= STRIDE:
            steps = 0
            v_current = v_pred
        vel[i + 1] = v_current
    return vel


def run_vel_only_filter(model, imu_data, timestamps, norm, device,
                         vel_init, start, end, lstm_state0):
    v = vel_init.copy().astype(np.float64)
    R_tcn = np.diag([0.61**2, 0.67**2, 0.25**2])
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
        v_pred, state = lstm_step(model, imu_data[k], norm, device, state)
        steps += 1
        if steps >= STRIDE:
            steps = 0
            S = P + R_tcn
            K = P @ np.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (np.eye(3) - K) @ P
        vel[i + 1] = v
    return vel


def run_strapdown_ekf_lstm(model, imu_data, accel, gyro, timestamps, norm, device,
                            ekf_warm, start, end, sigma_tcn, lstm_state0):
    ekf = copy.deepcopy(ekf_warm)
    R_tcn = np.eye(3) * sigma_tcn ** 2
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = ekf.s.v.copy()
    state = lstm_state0
    steps = 0
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[k], gyro[k], dt)
        v_pred, state = lstm_step(model, imu_data[k], norm, device, state)
        steps += 1
        if steps >= STRIDE:
            steps = 0
            ekf.update_velocity(v_pred, R_tcn)
        vel[i + 1] = ekf.s.v.copy()
    return vel


def run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, start, end):
    ekf = init_from_static(accel_static, gt_vel[start])
    n = end - start
    vel = np.zeros((n + 1, 3))
    vel[0] = gt_vel[start]
    for i in range(n):
        k = start + i
        dt = float(timestamps[k + 1] - timestamps[k])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[k], gyro[k], dt)
        ekf.update_velocity(gt_vel[k + 1])
        vel[i + 1] = ekf.s.v.copy()
    return vel


def evaluate_outage(df, model, norm, device, outage_s, n_static=200):
    timestamps = df["timestamp"].to_numpy()
    accel  = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro   = df[["gyro_x",  "gyro_y",  "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x","gt_vel_y","gt_vel_z"]].to_numpy()
    imu_data = df[["gyro_x","gyro_y","gyro_z","accel_x","accel_y","accel_z"]].to_numpy()

    accel_static = accel[:n_static].mean(axis=0)
    duration = timestamps[-1] - timestamps[0]
    outage_start_t = duration * 0.4
    outage_start = int(np.searchsorted(timestamps - timestamps[0], outage_start_t))
    outage_end   = min(len(df) - 1,
                       int(np.searchsorted(timestamps - timestamps[0], outage_start_t + outage_s)))

    ekf_warm = init_from_static(accel_static, gt_vel[0])
    for i in range(1, outage_start + 1):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf_warm.predict(accel[i], gyro[i], dt)
        ekf_warm.update_velocity(gt_vel[i])

    # Warm up LSTM over pre-outage data so it has full context at outage start
    print(f"  Warming LSTM over {outage_start} pre-outage samples...", end=" ", flush=True)
    _, lstm_state0 = warm_lstm(model, imu_data, norm, device, outage_start)
    print("done")

    ba_est = ekf_warm.s.ba.copy()
    vel_at_outage = gt_vel[outage_start]
    gt_outage = gt_vel[outage_start:outage_end + 1]

    def ferr(v):
        n = min(len(v), len(gt_outage)) - 1
        return float(np.linalg.norm(v[n] - gt_outage[n]))

    def merr(v):
        n = min(len(v), len(gt_outage))
        return float(np.mean(np.linalg.norm(v[:n] - gt_outage[:n], axis=1)))

    r = {"outage_s": outage_s}

    v_dr = run_dead_reckoning(accel, timestamps, ba_est, vel_at_outage, outage_start, outage_end)
    r["dr_final"], r["dr_mean"] = ferr(v_dr), merr(v_dr)

    v_lstm = run_standalone_lstm(model, imu_data, norm, device, vel_at_outage,
                                  outage_start, outage_end, lstm_state0)
    r["tcn_final"], r["tcn_mean"] = ferr(v_lstm), merr(v_lstm)

    for sig in SIGMA_TCN_SWEEP:
        v = run_strapdown_ekf_lstm(model, imu_data, accel, gyro, timestamps, norm, device,
                                    ekf_warm, outage_start, outage_end, sig, lstm_state0)
        key = f"nekf_{sig:.2f}"
        r[f"{key}_final"], r[f"{key}_mean"] = ferr(v), merr(v)

    v_vof = run_vel_only_filter(model, imu_data, timestamps, norm, device,
                                 vel_at_outage, outage_start, outage_end, lstm_state0)
    r["vof_final"], r["vof_mean"] = ferr(v_vof), merr(v_vof)

    v_gps = run_ekf_gps(accel, gyro, gt_vel, timestamps, accel_static, outage_start, outage_end)
    r["gps_final"], r["gps_mean"] = ferr(v_gps), merr(v_gps)

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60")
    args = parser.parse_args()

    outage_durations = [float(x) for x in args.outages.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, norm = load_lstm_v12(device)
    print(f"Loaded lstm_v12.pt: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Sigma sweep: {SIGMA_TCN_SWEEP}  |  sigma_process: {SIGMA_PROCESS}\n")

    df = pd.read_csv(SEQUENCES_DIR / args.seq / "imu_aligned.csv")
    print(f"Sequence: {args.seq}  |  {len(df)} samples\n")

    all_results = []
    for t_out in outage_durations:
        print(f"--- Outage {t_out}s ---")
        r = evaluate_outage(df, model, norm, device, t_out)
        all_results.append(r)

    cols   = ["dr", "tcn"] + [f"nekf_{s:.2f}" for s in SIGMA_TCN_SWEEP] + ["vof", "gps"]
    labels = ["DR", "LSTM-v12"] + [f"EKF σ={s}" for s in SIGMA_TCN_SWEEP] + ["VelFilter", "EKF+GPS"]

    print(f"\n{'Outage':>8}" + "".join(f"  {l:>12}" for l in labels))
    print("-" * (10 + 14 * len(cols)))
    for r in all_results:
        row = f"{r['outage_s']:>6}s"
        for c in cols:
            row += f"  {r.get(c+'_final', float('nan')):>10.3f}"
        print(row)

    r30 = next((r for r in all_results if r["outage_s"] == 30), None)
    if r30:
        print("\n--- Mean velocity error at 30s ---")
        for c, l in zip(cols, labels):
            print(f"  {l:>14}: {r30.get(c+'_mean', float('nan')):.3f} m/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.seq}_results.json"
    with open(out_path, "w") as f:
        json.dump({"seq": args.seq, "sigma_sweep": SIGMA_TCN_SWEEP,
                   "sigma_process_vof": SIGMA_PROCESS, "results": all_results}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
