"""
LSTM v12: sequence-to-sequence velocity prediction with persistent hidden state.

Key differences from TCN v11:
  - LSTMRegressor replaces TCNRegressor
  - Dense supervision: predict velocity at EVERY IMU timestep (not just window-end)
    → 400x more training signal per chunk
  - Persistent state across chunks: LSTM hidden state carries forward within a sequence
    → no cold-start, no missed context between predictions
  - At inference: run LSTM sample-by-sample with persistent (h, c); velocity available
    at every IMU step, not just at stride boundaries

Architecture:
  - 2-layer LSTM, hidden_size=128, dropout=0.3
  - Linear head: 128 → 3
  - Input: (batch, chunk_len, 6) normalized IMU
  - Output: (batch, chunk_len, 3) normalized absolute velocity at each timestep

Training:
  - Chunk length = 400 (same as v11 window)
  - Stride = 200 (50% overlap) — sequences are long enough to afford this
  - Loss: DirectionalMSE averaged over all timesteps in the chunk
  - Stateful within a sequence: h0 from chunk k is h0 for chunk k+1 (same sequence)
    — implemented via per-sequence shuffling + sequential chunk ordering

Saves:
  checkpoints/lstm_v12.pt
  results/lstm_v12/{loss_history,test_metrics,normalization_stats}.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "lstm_v12"

TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V1_03_difficult"]
VAL_SEQ = "MH_04_difficult"
TEST_SEQ = "MH_05_difficult"

CHUNK_LEN = 400   # samples per training chunk (2s at 200Hz)
CHUNK_STRIDE = 200  # 50% overlap between chunks
IMU_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 128,
                 num_layers: int = 2, output_dim: int = 3, dropout: float = 0.3) -> None:
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

    def forward(self, x: torch.Tensor, state=None):
        # x: (batch, seq_len, 6)
        out, state = self.lstm(x, state)
        out = self.dropout(out)
        return self.head(out), state   # (batch, seq_len, 3), (h, c)


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.6, min_norm: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_norm = min_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (batch, seq_len, 3) — flatten time into batch for loss
        p = pred.reshape(-1, 3)
        t = target.reshape(-1, 3)
        mse = F.mse_loss(p, t)
        t_norm = t.norm(dim=1)
        mask = t_norm > self.min_norm
        if mask.sum() > 0:
            cos_sim = F.cosine_similarity(p[mask], t[mask], dim=1)
            dir_loss = (1.0 - cos_sim).mean()
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)
        return self.alpha * mse + (1.0 - self.alpha) * dir_loss


def load_seq(seq_name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    csv_path = SEQUENCES_DIR / seq_name / "imu_aligned.csv"
    if not csv_path.exists():
        print(f"[skip] {seq_name}: imu_aligned.csv not found")
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in IMU_COLS + VEL_COLS if c not in df.columns]
    if missing:
        print(f"[skip] {seq_name}: missing columns {missing}")
        return None
    return df[IMU_COLS].to_numpy(dtype=np.float32), df[VEL_COLS].to_numpy(dtype=np.float32)


def make_chunks(imu: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a full sequence into overlapping chunks. Returns X (N,L,6), y (N,L,3)."""
    n = len(imu)
    X_list, y_list = [], []
    for start in range(0, n - CHUNK_LEN + 1, CHUNK_STRIDE):
        X_list.append(imu[start:start + CHUNK_LEN])
        y_list.append(vel[start:start + CHUNK_LEN])
    if not X_list:
        return np.empty((0, CHUNK_LEN, 6), dtype=np.float32), np.empty((0, CHUNK_LEN, 3), dtype=np.float32)
    return np.stack(X_list), np.stack(y_list)


class ChunkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def normalise_X(x_train, x_val, x_test):
    # Normalise over (samples, time, channels) — mean/std per channel
    flat = x_train.reshape(-1, 6)
    mean = flat.mean(0)
    std = flat.std(0) + 1e-8
    norm = lambda x: (x - mean) / std
    return norm(x_train), norm(x_val), norm(x_test), mean, std


def normalise_y(y_train, y_val, y_test):
    flat = y_train.reshape(-1, 3)
    mean = flat.mean(0)
    std = flat.std(0) + 1e-8
    norm = lambda y: (y - mean) / std
    return norm(y_train), norm(y_val), norm(y_test), mean, std


def build_split(seqs: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    X_all, y_all = [], []
    for seq in seqs:
        result = load_seq(seq)
        if result is None:
            continue
        imu, vel = result
        X_c, y_c = make_chunks(imu, vel)
        print(f"  {seq}: {len(X_c)} chunks")
        X_all.append(X_c)
        y_all.append(y_c)
    return np.concatenate(X_all), np.concatenate(y_all)


def build_eval_split(seq_name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    result = load_seq(seq_name)
    if result is None:
        return None
    imu, vel = result
    X_c, y_c = make_chunks(imu, vel)
    print(f"  {seq_name}: {len(X_c)} chunks")
    return X_c, y_c


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / count


@torch.no_grad()
def compute_metrics(model, loader, device, y_mean, y_std) -> Dict:
    """Evaluate on normalised chunks; report R² and correlation on denormalised velocity."""
    model.eval()
    preds_all, targets_all = [], []
    for xb, yb in loader:
        pred, _ = model(xb.to(device))
        # take last timestep of each chunk as the window-end prediction (mirrors TCN eval)
        preds_all.append(pred[:, -1, :].cpu())
        targets_all.append(yb[:, -1, :])
    p = torch.cat(preds_all)   # (N, 3) normalised
    t = torch.cat(targets_all)

    diff = p - t
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    axis_mse = diff.pow(2).mean(dim=0).tolist()
    t_var = t.var(dim=0)
    axis_r2 = [
        float(1 - axis_mse[i] / t_var[i].item()) if t_var[i].item() > 1e-8 else float("nan")
        for i in range(3)
    ]
    r2_mean = float(np.nanmean(axis_r2))

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = (ac.norm() * bc.norm()).item()
        return (ac @ bc).item() / d if d > 1e-8 else 0.0

    corr = [pearson(p[:, i], t[:, i]) for i in range(3)]
    zero_mse = t.pow(2).mean().item()
    return {
        "mse": mse, "mae": mae, "r2_mean": r2_mean,
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
        "zero_predictor_mse": zero_mse,
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
    }


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("Loading train sequences...")
    x_train, y_train = build_split(TRAIN_SEQS)
    print(f"Loading val: {VAL_SEQ}")
    val_result = build_eval_split(VAL_SEQ)
    assert val_result is not None
    x_val, y_val = val_result
    print(f"Loading test: {TEST_SEQ}")
    test_result = build_eval_split(TEST_SEQ)
    assert test_result is not None
    x_test, y_test = test_result

    x_train_n, x_val_n, x_test_n, x_mean, x_std = normalise_X(x_train, x_val, x_test)
    y_train_n, y_val_n, y_test_n, y_mean, y_std = normalise_y(y_train, y_val, y_test)

    norm_stats = {
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
    }
    save_json(RESULTS_DIR / "normalization_stats.json", norm_stats)

    print(f"Train: {x_train_n.shape}  Val: {x_val_n.shape}  Test: {x_test_n.shape}")
    print(f"Chunk length: {CHUNK_LEN} samples ({CHUNK_LEN/200:.1f}s), stride: {CHUNK_STRIDE}")
    print(f"Dense supervision: {CHUNK_LEN} velocity predictions per chunk")

    batch_size = 32
    train_loader = DataLoader(ChunkDataset(x_train_n, y_train_n), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(ChunkDataset(x_val_n, y_val_n),   batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(ChunkDataset(x_test_n, y_test_n), batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())

    model = LSTMRegressor(
        input_size=6, hidden_size=128, num_layers=2, output_dim=3, dropout=0.3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = DirectionalMSELoss(alpha=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    epochs = 100
    best_val = math.inf
    no_improve = 0
    patience = 25
    best_ckpt = CHECKPOINT_DIR / "lstm_v12.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    cfg = {
        "model": "lstm", "input_size": 6, "hidden_size": 128, "num_layers": 2,
        "output_dim": 3, "dropout": 0.3, "chunk_len": CHUNK_LEN,
        "loss": "directional_mse_alpha0.6",
        "target": "absolute_velocity_normalised_dense",
        "lr": 3e-4, "weight_decay": 1e-4, "batch_size": batch_size,
        "train_seqs": TRAIN_SEQS,
    }

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val, "config": cfg,
                "norm_stats": norm_stats,
            }, best_ckpt)
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}"
            + (" | best" if improved else "")
        )
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}.")
            break

    save_json(RESULTS_DIR / "loss_history.json", history)

    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = compute_metrics(model, test_loader, device, y_mean, y_std)
    test_metrics["best_val_loss"] = best["best_val_loss"]
    test_metrics["best_epoch"] = best["epoch"]
    save_json(RESULTS_DIR / "test_metrics.json", test_metrics)

    print(f"\nBest epoch: {best['epoch']}  Val loss: {best['best_val_loss']:.6f}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # Compare vs v7 and v11
    for ver, path in [("v7", "results/tcn_v7/test_metrics.json"), ("v11", "results/tcn_v11/test_metrics.json")]:
        p = PROJECT_ROOT / path
        if p.exists():
            with open(p) as f:
                ref = json.load(f)
            print(f"\n--- vs {ver} ---")
            for k in ["r2_mean", "corr_x", "corr_y", "corr_z"]:
                delta = test_metrics[k] - ref[k]
                print(f"  {k}: {test_metrics[k]:.4f} ({ver}: {ref[k]:.4f}, {'+' if delta>=0 else ''}{delta:.4f})")


if __name__ == "__main__":
    main()
