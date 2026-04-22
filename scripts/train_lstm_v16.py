"""
LSTM v16: hybrid loss — 50% per-step velocity-weighted MSE + 50% nav loss (10s outage).

Pure nav loss (v14/v15) risks mode collapse: a model that always predicts the current
filter velocity (low innovation) will produce low Kalman gain and stable-but-wrong outputs.
The per-step component (from v13) keeps individual predictions calibrated.

Loss:
  L = 0.5 * VelocityWeightedDirectionalMSELoss(per-step chunks)
    + 0.5 * mean(||v_filter - v_gt||) over 10s nav outage

Training alternates:
  - Per-step loss: computed on randomly sampled CHUNK_LEN=400 chunks (same as v13)
  - Nav loss: computed on OUTAGE_LEN=2000 outage with differentiable filter
  Both computed in the same forward pass (same model call).

Saves:
  checkpoints/lstm_v16.pt
  results/lstm_v16/{test_metrics,normalization_stats,loss_history}.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "lstm_v16"

TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V1_03_difficult"]
VAL_SEQ = "MH_04_difficult"
TEST_SEQ = "MH_05_difficult"

IMU_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]

CHUNK_LEN = 400
CHUNK_STRIDE = 200
OUTAGE_LEN = 2000       # 10s nav loss component
WARMUP_LEN = 1000
ACCUMULATE = 4
STRIDE = 25
SIGMA_PROCESS = 0.5
R_DIAG = [0.61**2, 0.67**2, 0.25**2]

ALPHA_STEP = 0.6        # per-step loss: weight between MSE and directional
NAV_WEIGHT = 0.5        # fraction of total loss from nav component
STEP_WEIGHT = 0.5

LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 20
STEPS_PER_EPOCH = 200
VAL_OUTAGE_LEN = 6000


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class VelocityWeightedDirectionalMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.6, min_norm: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_norm = min_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v_norm = target.norm(dim=-1)
        weights = v_norm / (v_norm.mean() + 1e-8)
        diff_sq = (pred - target).pow(2).mean(dim=-1)
        weighted_mse = (weights * diff_sq).mean()

        p_flat = pred.reshape(-1, 3)
        t_flat = target.reshape(-1, 3)
        mask = t_flat.norm(dim=1) > self.min_norm
        if mask.sum() > 0:
            cos_sim = F.cosine_similarity(p_flat[mask], t_flat[mask], dim=1)
            dir_loss = (1.0 - cos_sim).mean()
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)

        return self.alpha * weighted_mse + (1.0 - self.alpha) * dir_loss


def load_norm_stats() -> Dict:
    p = PROJECT_ROOT / "results" / "lstm_v13" / "normalization_stats.json"
    with open(p) as f:
        ns = json.load(f)
    return {
        "x_mean": np.array(ns["x_mean"], dtype=np.float32),
        "x_std":  np.array(ns["x_std"],  dtype=np.float32),
        "y_mean": np.array(ns["y_mean"], dtype=np.float32),
        "y_std":  np.array(ns["y_std"],  dtype=np.float32),
    }


def load_seq_raw(seq_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    csv_path = SEQUENCES_DIR / seq_name / "imu_aligned.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in IMU_COLS + VEL_COLS if c not in df.columns]
    if missing:
        return None
    return df[IMU_COLS].to_numpy(dtype=np.float32), df[VEL_COLS].to_numpy(dtype=np.float32)


def normalize_seq(imu: np.ndarray, vel: np.ndarray, ns: Dict) -> Tuple[np.ndarray, np.ndarray]:
    return (imu - ns["x_mean"]) / ns["x_std"], (vel - ns["y_mean"]) / ns["y_std"]


def run_differentiable_filter(
    model: LSTMRegressor,
    imu_tensor: torch.Tensor,
    vel_init: torch.Tensor,
    lstm_state: Tuple,
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> torch.Tensor:
    T = imu_tensor.shape[0]
    v = vel_init.detach().clone()
    P = torch.eye(3, device=device) * 0.5
    R_tcn = torch.diag(torch.tensor(R_DIAG, dtype=torch.float32, device=device))
    Q_rate = SIGMA_PROCESS ** 2 * torch.eye(3, device=device)
    I3 = torch.eye(3, device=device)
    dt = 0.005

    state = lstm_state
    steps_since_update = 0
    vel_history: List[torch.Tensor] = [v.unsqueeze(0)]

    for t in range(T):
        P = P + Q_rate * dt
        x_t = imu_tensor[t:t + 1].unsqueeze(0)
        y_norm, state = model(x_t, state)
        v_pred = y_norm[0, 0] * y_std_t + y_mean_t

        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            S = P + R_tcn
            K = P @ torch.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (I3 - K) @ P

        vel_history.append(v.unsqueeze(0))

    return torch.cat(vel_history, dim=0)


def warmup_lstm(
    model: LSTMRegressor,
    imu_norm: np.ndarray,
    outage_start: int,
    device: torch.device,
) -> Optional[Tuple]:
    warmup_start = max(0, outage_start - WARMUP_LEN)
    with torch.no_grad():
        state = None
        for k in range(warmup_start, outage_start):
            x = torch.tensor(imu_norm[k:k + 1][None], dtype=torch.float32, device=device)
            _, state = model(x, state)
    if state is None:
        return None
    return (state[0].detach(), state[1].detach())


def train_step(
    model: LSTMRegressor,
    seqs_norm: List[Tuple[np.ndarray, np.ndarray]],
    chunk_data: Tuple[np.ndarray, np.ndarray],
    optimizer: torch.optim.Optimizer,
    step_criterion: VelocityWeightedDirectionalMSELoss,
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    total_step_loss = 0.0
    total_nav_loss = 0.0

    x_chunks, y_chunks = chunk_data
    n_chunks = len(x_chunks)

    for _ in range(ACCUMULATE):
        # --- Per-step loss component (v13-style, random chunks) ---
        chunk_idx = rng.integers(0, n_chunks, size=32)
        xb = torch.tensor(x_chunks[chunk_idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_chunks[chunk_idx], dtype=torch.float32, device=device)
        pred, _ = model(xb)
        step_loss = STEP_WEIGHT * step_criterion(pred, yb) / ACCUMULATE
        step_loss.backward()
        total_step_loss += step_loss.item()

        # --- Nav loss component (differentiable filter) ---
        seq_idx = rng.integers(0, len(seqs_norm))
        imu_norm, vel_norm = seqs_norm[seq_idx]
        T = len(imu_norm)
        min_start = WARMUP_LEN
        max_start = T - OUTAGE_LEN - 1
        if max_start <= min_start:
            continue
        outage_start = int(rng.integers(min_start, max_start))

        state = warmup_lstm(model, imu_norm, outage_start, device)
        outage_imu = torch.tensor(
            imu_norm[outage_start:outage_start + OUTAGE_LEN],
            dtype=torch.float32, device=device)
        outage_vel_norm = torch.tensor(
            vel_norm[outage_start:outage_start + OUTAGE_LEN],
            dtype=torch.float32, device=device)

        vel_init = outage_vel_norm[0] * y_std_t + y_mean_t
        vel_filter = run_differentiable_filter(
            model, outage_imu, vel_init, state, device, y_mean_t, y_std_t)

        gt_denorm = outage_vel_norm * y_std_t + y_mean_t
        nav_loss = NAV_WEIGHT * (vel_filter[1:] - gt_denorm).norm(dim=-1).mean() / ACCUMULATE
        nav_loss.backward()
        total_nav_loss += nav_loss.item()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_step_loss, total_nav_loss


def make_chunks(imu_norm: np.ndarray, vel_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = len(imu_norm)
    X_list, y_list = [], []
    for start in range(0, T - CHUNK_LEN + 1, CHUNK_STRIDE):
        X_list.append(imu_norm[start:start + CHUNK_LEN])
        y_list.append(vel_norm[start:start + CHUNK_LEN])
    if not X_list:
        return np.empty((0, CHUNK_LEN, 6)), np.empty((0, CHUNK_LEN, 3))
    return np.stack(X_list), np.stack(y_list)


@torch.no_grad()
def validate(
    model: LSTMRegressor,
    val_norm: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> Tuple[float, float]:
    model.eval()
    imu_norm, vel_norm = val_norm
    T = len(imu_norm)
    outage_start = int(T * 0.4)
    outage_len = min(VAL_OUTAGE_LEN, T - outage_start - 1)

    state = None
    for k in range(outage_start):
        x = torch.tensor(imu_norm[k:k + 1][None], dtype=torch.float32, device=device)
        _, state = model(x, state)

    v = torch.tensor(vel_norm[outage_start], dtype=torch.float32, device=device) * y_std_t + y_mean_t
    P = torch.eye(3, device=device) * 0.5
    R_tcn = torch.diag(torch.tensor(R_DIAG, dtype=torch.float32, device=device))
    Q_rate = SIGMA_PROCESS ** 2 * torch.eye(3, device=device)
    I3 = torch.eye(3, device=device)
    gt = (torch.tensor(vel_norm[outage_start:outage_start + outage_len],
                        dtype=torch.float32, device=device) * y_std_t + y_mean_t)

    errors = []
    steps_since_update = 0
    for t in range(outage_len):
        P = P + Q_rate * 0.005
        x = torch.tensor(imu_norm[outage_start + t:outage_start + t + 1][None],
                          dtype=torch.float32, device=device)
        y_norm_t, state = model(x, state)
        v_pred = y_norm_t[0, 0] * y_std_t + y_mean_t
        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            S = P + R_tcn
            K = P @ torch.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (I3 - K) @ P
        errors.append((v - gt[t]).norm().item())

    return float(np.mean(errors)), float(errors[-1])


@torch.no_grad()
def compute_chunk_metrics(
    model: LSTMRegressor,
    imu_norm: np.ndarray,
    vel_norm: np.ndarray,
    device: torch.device,
) -> Dict:
    model.eval()
    T = len(imu_norm)
    preds, targets = [], []
    for start in range(0, T - CHUNK_LEN + 1, CHUNK_STRIDE):
        xb = torch.tensor(imu_norm[start:start + CHUNK_LEN][None],
                           dtype=torch.float32, device=device)
        pred, _ = model(xb)
        preds.append(pred[0, -1, :].cpu().numpy())
        targets.append(vel_norm[start + CHUNK_LEN - 1])

    p, t = np.stack(preds), np.stack(targets)
    diff = p - t
    mse = float(np.mean(diff ** 2))
    axis_mse = [float(np.mean(diff[:, i] ** 2)) for i in range(3)]
    t_var = np.var(t, axis=0)
    axis_r2 = [float(1 - axis_mse[i] / t_var[i]) if t_var[i] > 1e-8 else float("nan")
               for i in range(3)]

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        return float((ac * bc).sum() / d) if d > 1e-8 else 0.0

    corr = [pearson(p[:, i], t[:, i]) for i in range(3)]
    return {
        "mse": mse, "mae": float(np.mean(np.abs(diff))),
        "r2_mean": float(np.nanmean(axis_r2)),
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ns = load_norm_stats()
    y_mean_t = torch.tensor(ns["y_mean"], dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(ns["y_std"],  dtype=torch.float32, device=device)

    print("Loading training sequences...")
    seqs_norm: List[Tuple[np.ndarray, np.ndarray]] = []
    all_chunks_X, all_chunks_y = [], []
    for seq in TRAIN_SEQS:
        raw = load_seq_raw(seq)
        if raw is None:
            continue
        imu_n, vel_n = normalize_seq(raw[0], raw[1], ns)
        seqs_norm.append((imu_n, vel_n))
        cx, cy = make_chunks(imu_n, vel_n)
        all_chunks_X.append(cx)
        all_chunks_y.append(cy)
        print(f"  {seq}: {len(imu_n)} samples, {len(cx)} chunks")

    chunk_data = (np.concatenate(all_chunks_X), np.concatenate(all_chunks_y))
    print(f"Total chunks: {len(chunk_data[0])}")

    val_raw = load_seq_raw(VAL_SEQ); assert val_raw is not None
    val_norm = normalize_seq(val_raw[0], val_raw[1], ns)
    test_raw = load_seq_raw(TEST_SEQ); assert test_raw is not None
    test_norm = normalize_seq(test_raw[0], test_raw[1], ns)

    model = LSTMRegressor().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss: {STEP_WEIGHT}×per-step(v13-weighted) + {NAV_WEIGHT}×nav(10s outage)")

    step_criterion = VelocityWeightedDirectionalMSELoss(alpha=ALPHA_STEP)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    rng = np.random.default_rng(42)
    best_val, no_improve = math.inf, 0
    best_ckpt = CHECKPOINT_DIR / "lstm_v16.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    norm_stats_out = {k: v.tolist() for k, v in ns.items()}
    save_json(RESULTS_DIR / "normalization_stats.json", norm_stats_out)

    cfg = {
        "model": "lstm", "hidden_size": 128, "num_layers": 2, "dropout": 0.3,
        "loss": f"hybrid_{STEP_WEIGHT}x_vel_weighted_step_{NAV_WEIGHT}x_nav_10s",
        "outage_len_s": OUTAGE_LEN / 200, "warmup_len_s": WARMUP_LEN / 200,
        "accumulate": ACCUMULATE, "lr": LR, "train_seqs": TRAIN_SEQS,
    }

    history: Dict[str, List] = {
        "step_loss": [], "nav_loss": [], "val_mean_err": [], "val_final_err": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        step_total, nav_total = 0.0, 0.0
        for _ in range(STEPS_PER_EPOCH):
            s, n = train_step(model, seqs_norm, chunk_data, optimizer, step_criterion,
                               device, y_mean_t, y_std_t, rng)
            step_total += s
            nav_total += n
        step_loss = step_total / STEPS_PER_EPOCH
        nav_loss  = nav_total  / STEPS_PER_EPOCH

        val_mean, val_final = validate(model, val_norm, device, y_mean_t, y_std_t)
        scheduler.step()

        history["step_loss"].append(step_loss)
        history["nav_loss"].append(nav_loss)
        history["val_mean_err"].append(val_mean)
        history["val_final_err"].append(val_final)

        improved = val_mean < best_val
        if improved:
            best_val, no_improve = val_mean, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mean_err": best_val, "config": cfg, "norm_stats": norm_stats_out,
            }, best_ckpt)
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | step {step_loss:.4f} | nav {nav_loss:.4f} | "
              f"val_mean {val_mean:.4f} | val_final {val_final:.4f}"
              + (" | best" if improved else ""))

        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch}.")
            break

    save_json(RESULTS_DIR / "loss_history.json", history)

    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    chunk_metrics = compute_chunk_metrics(model, test_norm[0], test_norm[1], device)
    test_val_mean, test_val_final = validate(model, test_norm, device, y_mean_t, y_std_t)

    test_metrics = {
        **chunk_metrics,
        "nav_val_mean_err": test_val_mean,
        "nav_val_final_err": test_val_final,
        "best_val_mean_err": best["best_val_mean_err"],
        "best_epoch": best["epoch"],
    }
    save_json(RESULTS_DIR / "test_metrics.json", test_metrics)

    print(f"\nBest epoch: {best['epoch']}  Val mean err: {best['best_val_mean_err']:.4f} m/s")
    print("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    for ver, path in [("v13", "results/lstm_v13/test_metrics.json")]:
        p = PROJECT_ROOT / path
        if p.exists():
            with open(p) as f:
                ref = json.load(f)
            print(f"\n--- vs {ver} ---")
            for k in ["r2_mean", "corr_x", "corr_y", "corr_z"]:
                if k in ref:
                    delta = chunk_metrics[k] - ref[k]
                    print(f"  {k}: {chunk_metrics[k]:.4f} ({ver}: {ref[k]:.4f}, "
                          f"{'+' if delta >= 0 else ''}{delta:.4f})")


if __name__ == "__main__":
    main()
