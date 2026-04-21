"""
TCN v10: noise-only augmentation (no rotation, no reflection).

Hypothesis: v7 train loss (0.88) << val loss (1.29) — the model is overfitting.
Mild Gaussian noise in normalized space should reduce this gap without
destroying heading-specific patterns. v9's yaw rotation failed because
EuRoC test sequences have heading-specific dynamics that the model was
usefully exploiting; rotating augmented away that information.

Augmentation applied per-window during training only:
  - Gaussian noise σ=0.05 on all 6 normalized IMU channels (5% of unit std)
  - No rotation, no reflection

Changes from v7:
  - AugmentedDataset with noise-only augmentation
  - Val/test loaders unchanged

Saves:
  checkpoints/tcn_v10.pt
  results/tcn_v10/{loss_history,test_metrics}.json
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
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "tcn_v10"

TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V1_03_difficult"]
VAL_SEQ = "MH_04_difficult"
TEST_SEQ = "MH_05_difficult"

WINDOW_SIZE = 200
STRIDE = 25
IMU_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]

# Noise augmentation — applied in normalized space (std≈1 per channel)
SIGMA_NOISE = 0.05   # 5% of unit std across all IMU channels


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_window(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to normalized IMU window. Velocity label unchanged."""
    x = x.copy()
    x += np.random.randn(*x.shape).astype(np.float32) * SIGMA_NOISE
    return x, y


class AugmentedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x, y = augment_window(self.X[idx], self.y[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.6, min_norm: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_norm = min_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        t_norm = target.norm(dim=1, keepdim=True)
        mask = t_norm.squeeze(1) > self.min_norm
        if mask.sum() > 0:
            cos_sim = F.cosine_similarity(pred[mask], target[mask], dim=1)
            dir_loss = (1.0 - cos_sim).mean()
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)
        return self.alpha * mse + (1.0 - self.alpha) * dir_loss


def build_windows_for_seq(seq_name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    csv_path = SEQUENCES_DIR / seq_name / "imu_aligned.csv"
    if not csv_path.exists():
        print(f"[skip] {seq_name}: imu_aligned.csv not found")
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in IMU_COLS + VEL_COLS if c not in df.columns]
    if missing:
        print(f"[skip] {seq_name}: missing columns {missing}")
        return None
    imu = df[IMU_COLS].to_numpy(dtype=np.float32)
    vel = df[VEL_COLS].to_numpy(dtype=np.float32)
    n = len(df)
    X_list, y_list = [], []
    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE - 1
        X_list.append(imu[start:end + 1])
        y_list.append(vel[end])
    if not X_list:
        return None
    return np.stack(X_list), np.stack(y_list)


def load_split(seqs: list[str]) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    X_all, y_all, per_seq_y = [], [], []
    for seq in seqs:
        result = build_windows_for_seq(seq)
        if result is None:
            continue
        X, y = result
        print(f"  {seq}: {len(X)} windows")
        X_all.append(X)
        y_all.append(y)
        per_seq_y.append(y)
    return np.concatenate(X_all), np.concatenate(y_all), per_seq_y


def normalise_X(x_train, x_val, x_test):
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean, std


def normalise_y(y_train, y_val, y_test):
    mean = y_train.mean(axis=0, keepdims=True)
    std = y_train.std(axis=0, keepdims=True) + 1e-8
    return (y_train - mean) / std, (y_val - mean) / std, (y_test - mean) / std, mean, std


def build_balanced_sampler(per_seq_y: list[np.ndarray]) -> WeightedRandomSampler:
    weights = []
    for y_seq in per_seq_y:
        n = len(y_seq)
        weights.extend([1.0 / n] * n)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loader(x, y, batch_size, sampler=None, shuffle=False, augment=False) -> DataLoader:
    if augment:
        ds = AugmentedDataset(x, y)
    else:
        ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(
        ds, batch_size=batch_size,
        sampler=sampler, shuffle=(shuffle and sampler is None),
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / count


@torch.no_grad()
def compute_metrics(model, loader, device) -> Dict:
    model.eval()
    preds_all, targets_all = [], []
    for xb, yb in loader:
        preds_all.append(model(xb.to(device)).cpu())
        targets_all.append(yb)
    p = torch.cat(preds_all)
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    print("Loading train sequences...")
    x_train, y_train, per_seq_y = load_split(TRAIN_SEQS)
    print(f"Loading val: {VAL_SEQ}")
    val_result = build_windows_for_seq(VAL_SEQ)
    assert val_result is not None
    x_val, y_val = val_result
    print(f"  {VAL_SEQ}: {len(x_val)} windows")
    print(f"Loading test: {TEST_SEQ}")
    test_result = build_windows_for_seq(TEST_SEQ)
    assert test_result is not None
    x_test, y_test = test_result
    print(f"  {TEST_SEQ}: {len(x_test)} windows")

    # Normalise on raw (pre-augmentation) train stats
    x_train_n, x_val_n, x_test_n, x_mean, x_std = normalise_X(x_train, x_val, x_test)
    y_train_n, y_val_n, y_test_n, y_mean, y_std = normalise_y(y_train, y_val, y_test)

    norm_stats = {
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "normalization_stats.json", norm_stats)

    print(f"Train: {x_train_n.shape}  Val: {x_val_n.shape}  Test: {x_test_n.shape}")
    print(f"Augmentation: Gaussian noise σ={SIGMA_NOISE} on normalized IMU (no rotation)")

    sampler = build_balanced_sampler(per_seq_y)
    batch_size = 64
    # Train loader uses augmented dataset; val/test do not
    train_loader = make_loader(x_train_n, y_train_n, batch_size, sampler=sampler, augment=True)
    val_loader = make_loader(x_val_n, y_val_n, batch_size)
    test_loader = make_loader(x_test_n, y_test_n, batch_size)

    model = TCNRegressor(
        input_channels=6, output_dim=3,
        channel_sizes=[16, 32, 32], kernel_size=3, dropout=0.3,
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
    best_ckpt = CHECKPOINT_DIR / "tcn_v10.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    cfg = {
        "input_channels": 6, "output_dim": 3,
        "channel_sizes": [16, 32, 32], "kernel_size": 3, "dropout": 0.3,
        "loss": "directional_mse_alpha0.6",
        "target": "absolute_velocity_normalised",
        "augmentation": {"noise_sigma": SIGMA_NOISE},
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
    test_metrics = compute_metrics(model, test_loader, device)
    test_metrics["best_val_loss"] = best["best_val_loss"]
    test_metrics["best_epoch"] = best["epoch"]
    save_json(RESULTS_DIR / "test_metrics.json", test_metrics)

    print(f"\nBest epoch: {best['epoch']}  Val loss: {best['best_val_loss']:.6f}")
    print("Test metrics (on normalised y):")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
