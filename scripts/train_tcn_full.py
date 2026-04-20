from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split_data(split_dir: Path) -> Tuple[np.ndarray, ...]:
    x_train = np.load(split_dir / "X_train.npy")
    y_train = np.load(split_dir / "y_train.npy")
    x_val = np.load(split_dir / "X_val.npy")
    y_val = np.load(split_dir / "y_val.npy")
    x_test = np.load(split_dir / "X_test.npy")
    y_test = np.load(split_dir / "y_test.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        preds = model(xb)
        loss = criterion(preds, yb)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        preds = model(xb)

        preds_all.append(preds.cpu())
        targets_all.append(yb.cpu())

    preds_cat = torch.cat(preds_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)

    diff = preds_cat - targets_cat

    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()

    axis_mse = torch.mean(diff ** 2, dim=0).tolist()
    axis_mae = torch.mean(torch.abs(diff), dim=0).tolist()

    # R² per axis: 1 - MSE / Var(y). Negative means worse than zero predictor.
    target_var = targets_cat.var(dim=0)
    axis_r2 = [
        (1.0 - axis_mse[i] / target_var[i].item()) if target_var[i].item() > 1e-8 else float("nan")
        for i in range(3)
    ]
    r2_mean = float(sum(v for v in axis_r2 if not (v != v)) / max(1, sum(1 for v in axis_r2 if v == v)))

    # Per-axis Pearson correlation
    def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = (a_c.norm() * b_c.norm()).item()
        return (a_c @ b_c).item() / denom if denom > 1e-8 else 0.0

    axis_corr = [pearson(preds_cat[:, i], targets_cat[:, i]) for i in range(3)]

    # Zero-predictor baseline MSE (= variance of targets)
    zero_mse = torch.mean(targets_cat ** 2).item()

    return {
        "mse": mse,
        "mae": mae,
        "r2_mean": r2_mean,
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": axis_corr[0], "corr_y": axis_corr[1], "corr_z": axis_corr[2],
        "zero_predictor_mse": zero_mse,
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
        "mae_x": axis_mae[0], "mae_y": axis_mae[1], "mae_z": axis_mae[2],
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    set_seed(42)

    split_dir = PROJECT_ROOT / "data" / "splits"
    if not split_dir.exists():
        split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    results_dir = PROJECT_ROOT / "results" / "tcn_multi"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = 100
    batch_size = 32
    learning_rate = 3e-4

    x_train, y_train, x_val, y_val, x_test, y_test = load_split_data(split_dir)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Val shape:   {x_val.shape}, {y_val.shape}")
    print(f"Test shape:  {x_test.shape}, {y_test.shape}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model = TCNRegressor(
        input_channels=6,
        output_dim=3,
        channel_sizes=[16, 32, 32],
        kernel_size=3,
        dropout=0.3,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = math.inf
    best_checkpoint_path = checkpoint_dir / "tcn_multi.pt"
    early_stop_patience = 20
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "input_channels": 6,
                        "output_dim": 3,
                        "channel_sizes": [16, 32, 32],
                        "kernel_size": 3,
                        "dropout": 0.3,
                        "weight_decay": 1e-4,
                    },
                },
                best_checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f}"
            + (" | saved best" if improved else "")
        )

        scheduler.step(val_loss)

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping: no val improvement for {early_stop_patience} epochs.")
            break

    save_json(results_dir / "loss_history.json", history)

    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = compute_metrics(model, test_loader, device)
    test_metrics["best_val_loss"] = best_ckpt["best_val_loss"]
    test_metrics["best_epoch"] = best_ckpt["epoch"]

    save_json(results_dir / "test_metrics.json", test_metrics)

    print("\nBest checkpoint summary")
    print(f"Best epoch: {best_ckpt['epoch']}")
    print(f"Best val loss: {best_ckpt['best_val_loss']:.8f}")

    print("\nTest metrics")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()