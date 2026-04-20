#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "tcn_baseline"
    loss_history_path = results_dir / "loss_history.json"
    output_path = results_dir / "loss_curve.png"

    if not loss_history_path.exists():
        raise FileNotFoundError(f"Missing file: {loss_history_path}")

    with open(loss_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    if len(train_loss) != len(val_loss):
        raise ValueError("train_loss and val_loss must have the same length")

    epochs = list(range(1, len(train_loss) + 1))

    best_val_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i]) + 1
    best_val_loss = val_loss[best_val_epoch - 1]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.axvline(best_val_epoch, linestyle="--", label=f"Best Val Epoch ({best_val_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("TCN Baseline Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved loss curve to: {output_path}")
    print(f"Best validation epoch: {best_val_epoch}")
    print(f"Best validation loss: {best_val_loss:.8f}")


if __name__ == "__main__":
    main()