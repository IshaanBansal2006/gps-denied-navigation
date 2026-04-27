"""
TCN v2: directional loss + larger model + balanced training.

Changes from v1 (train_tcn_full.py):
  - Loss: 0.6*MSE + 0.4*directional (cosine dissimilarity)
    Forces model to learn direction of velocity change, not just magnitude
  - Model: [64, 128, 256, 256] (4x more parameters)
  - Batch size: 64, lr: 3e-4
  - Sequence-balanced sampling: equal weight per sequence regardless of size

Saves:
  checkpoints/tcn_v2.pt
  results/tcn_v2/{loss_history,test_metrics}.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "tcn_v2"


# Sequences per split (must match build_dataset.py)
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
    """
    Combined MSE + cosine dissimilarity loss.
    L = alpha * MSE + (1-alpha) * mean(1 - cosine_sim)

    The cosine term is only computed for targets with |y| > min_norm to
    avoid division by near-zero vectors.
    """
    def __init__(self, alpha: float = 0.6, min_norm: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_norm = min_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        # Cosine dissimilarity only where target has meaningful magnitude
        t_norm = target.norm(dim=1, keepdim=True)
        mask = (t_norm.squeeze(1) > self.min_norm)
        if mask.sum() > 0:
            cos_sim = F.cosine_similarity(pred[mask], target[mask], dim=1)
            dir_loss = (1.0 - cos_sim).mean()
        else:
            dir_loss = torch.tensor(0.0, device=pred.device)

        return self.alpha * mse + (1.0 - self.alpha) * dir_loss


def build_balanced_sampler(y_train_per_seq: list[np.ndarray]) -> WeightedRandomSampler:
    """Each sequence gets equal total weight regardless of its window count."""
    weights = []
    for y_seq in y_train_per_seq:
        n = len(y_seq)
        weights.extend([1.0 / n] * n)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def load_split_data() -> Tuple[np.ndarray, ...]:
    x_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    x_val = np.load(SPLIT_DIR / "X_val.npy")
    y_val = np.load(SPLIT_DIR / "y_val.npy")
    x_test = np.load(SPLIT_DIR / "X_test.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_per_seq_y(norm_stats: dict) -> list[np.ndarray]:
    """Load y arrays per training sequence for balanced sampler."""
    result = []
    for seq in TRAIN_SEQS:
        y_path = SEQUENCES_DIR / seq / "y_delta_v.npy"
        if y_path.exists():
            result.append(np.load(y_path))
    return result


def make_loader(x, y, batch_size, sampler=None, shuffle=False) -> DataLoader:
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

    x_train, y_train, x_val, y_val, x_test, y_test = load_split_data()

    # Per-sequence y arrays for balanced sampler
    per_seq_y = load_per_seq_y(json.load(open(SPLIT_DIR / "normalization_stats.json")))
    sampler = build_balanced_sampler(per_seq_y) if per_seq_y else None

    batch_size = 64
    train_loader = make_loader(x_train, y_train, batch_size, sampler=sampler)
    val_loader = make_loader(x_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)

    print(f"Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    model = TCNRegressor(
        input_channels=6,
        output_dim=3,
        channel_sizes=[64, 128, 256, 256],
        kernel_size=3,
        dropout=0.2,
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
    best_ckpt = CHECKPOINT_DIR / "tcn_v2.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    cfg = {
        "input_channels": 6, "output_dim": 3,
        "channel_sizes": [64, 128, 256, 256],
        "kernel_size": 3, "dropout": 0.2,
        "loss": "directional_mse_alpha0.6",
        "lr": 3e-4, "weight_decay": 1e-4, "batch_size": batch_size,
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
    print("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
