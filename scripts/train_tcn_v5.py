"""
TCN v5: EKF-smoothed labels.

Hypothesis: The EKF (run in full-GPS mode offline) produces physically
consistent, noise-filtered velocity estimates — a better smoother than
Savitzky-Golay because it incorporates the IMU motion model rather than
applying a generic polynomial fit.

Label generation:
  - For each sequence, run the 15-state EKF with Leica velocity at every step
  - Use EKF-estimated velocity to compute delta_v per window
  - X windows unchanged (same raw IMU input as all prior runs)

Changes from v3:
  - Labels: EKF velocity replaces raw Leica finite-difference velocity
  - Model/loss/sampler: identical to v3/v4 ([16,32,32], directional loss alpha=0.6)

Saves:
  checkpoints/tcn_v5.pt
  results/tcn_v5/{loss_history,test_metrics}.json
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor
from src.filters.ekf import init_from_static

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "tcn_v5"

WINDOW_SIZE = 200
STRIDE = 25
N_STATIC = 200  # samples used to estimate initial accel bias

TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium"]
VAL_SEQ = "MH_04_difficult"
TEST_SEQ = "MH_05_difficult"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            dir_loss = (1.0 - F.cosine_similarity(pred[mask], target[mask], dim=1)).mean()
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)
        return self.alpha * mse + (1.0 - self.alpha) * dir_loss


def run_ekf_on_sequence(seq_name: str) -> np.ndarray:
    """Run full-GPS EKF over a sequence, return smoothed velocity (N, 3)."""
    df = pd.read_csv(SEQUENCES_DIR / seq_name / "imu_aligned.csv")
    timestamps = df["timestamp"].to_numpy()
    accel  = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    gyro   = df[["gyro_x",  "gyro_y",  "gyro_z"]].to_numpy()
    gt_vel = df[["gt_vel_x","gt_vel_y","gt_vel_z"]].to_numpy()

    accel_static = accel[:N_STATIC].mean(axis=0)
    ekf = init_from_static(accel_static, gt_vel[0])

    n = len(df)
    vel_ekf = np.zeros((n, 3), dtype=np.float32)
    vel_ekf[0] = gt_vel[0]

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0 or dt > 0.05:
            dt = 0.005
        ekf.predict(accel[i], gyro[i], dt)
        ekf.update_velocity(gt_vel[i])
        vel_ekf[i] = ekf.s.v

    return vel_ekf


def build_ekf_y(seq_name: str) -> np.ndarray:
    """Compute delta_v labels from EKF-smoothed velocity."""
    vel_ekf = run_ekf_on_sequence(seq_name)
    n = len(vel_ekf)
    y_list = []
    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE - 1
        y_list.append(vel_ekf[end] - vel_ekf[start])
    return np.array(y_list, dtype=np.float32)


def load_data() -> Tuple[np.ndarray, ...]:
    x_train = np.load(SPLIT_DIR / "X_train.npy")
    x_val   = np.load(SPLIT_DIR / "X_val.npy")
    x_test  = np.load(SPLIT_DIR / "X_test.npy")

    print("Running EKF on training sequences...")
    y_train = np.concatenate([build_ekf_y(s) for s in TRAIN_SEQS], axis=0)
    print("Running EKF on val/test sequences...")
    y_val  = build_ekf_y(VAL_SEQ)
    y_test = build_ekf_y(TEST_SEQ)

    assert len(x_train) == len(y_train), f"Train mismatch: {len(x_train)} vs {len(y_train)}"
    assert len(x_val)   == len(y_val),   f"Val mismatch: {len(x_val)} vs {len(y_val)}"
    assert len(x_test)  == len(y_test),  f"Test mismatch: {len(x_test)} vs {len(y_test)}"
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_balanced_sampler(seqs: list[str]) -> WeightedRandomSampler:
    weights = []
    for seq in seqs:
        n = len(build_ekf_y(seq))
        weights.extend([1.0 / n] * n)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loader(x, y, batch_size, sampler=None, shuffle=False) -> DataLoader:
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=(shuffle and sampler is None),
                      num_workers=0, pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        if optimizer:
            optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        if optimizer:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    diff = p - t
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    axis_mse = diff.pow(2).mean(dim=0).tolist()
    t_var = t.var(dim=0)
    axis_r2 = [
        float(1 - axis_mse[i] / t_var[i].item()) if t_var[i].item() > 1e-8 else float("nan")
        for i in range(3)
    ]

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = (ac.norm() * bc.norm()).item()
        return (ac @ bc).item() / d if d > 1e-8 else 0.0

    corr = [pearson(p[:, i], t[:, i]) for i in range(3)]
    return {
        "mse": mse, "mae": mae,
        "r2_mean": float(np.nanmean(axis_r2)),
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
        "zero_predictor_mse": t.pow(2).mean().item(),
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

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    print(f"Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")
    print(f"y_train std per axis: {y_train.std(axis=0).round(4)}")

    # Balanced sampler: reuse cached EKF y arrays (already computed above)
    # Simple weight approach: equal weight per sequence
    seq_sizes = [len(build_ekf_y(s)) for s in TRAIN_SEQS]
    weights = []
    for n in seq_sizes:
        weights.extend([1.0 / n] * n)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    batch_size = 64
    train_loader = make_loader(x_train, y_train, batch_size, sampler=sampler)
    val_loader   = make_loader(x_val,   y_val,   batch_size)
    test_loader  = make_loader(x_test,  y_test,  batch_size)

    model = TCNRegressor(
        input_channels=6, output_dim=3,
        channel_sizes=[16, 32, 32], kernel_size=3, dropout=0.3,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DirectionalMSELoss(alpha=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    best_val, no_improve = math.inf, 0
    best_ckpt = CHECKPOINT_DIR / "tcn_v5.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, 101):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val, no_improve = val_loss, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_val_loss": best_val,
                        "config": {"channel_sizes": [16,32,32], "loss": "directional_mse_alpha0.6",
                                   "label_smoothing": "ekf_full_gps"}}, best_ckpt)
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}"
              + (" | best" if improved else ""))
        if no_improve >= 25:
            print(f"Early stop at epoch {epoch}.")
            break

    save_json(RESULTS_DIR / "loss_history.json", history)
    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    metrics = compute_metrics(model, test_loader, device)
    metrics["best_val_loss"] = best["best_val_loss"]
    metrics["best_epoch"] = best["epoch"]
    save_json(RESULTS_DIR / "test_metrics.json", metrics)

    print(f"\nBest epoch: {best['epoch']}  Val loss: {best['best_val_loss']:.6f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
