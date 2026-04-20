#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor


X_TRAIN_PATH = Path("data/processed/splits/X_train.npy")
Y_TRAIN_PATH = Path("data/processed/splits/y_train.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUBSET_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 75
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)

    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {X_TRAIN_PATH}")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {Y_TRAIN_PATH}")

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    X_subset = X_train[:SUBSET_SIZE]
    y_subset = y_train[:SUBSET_SIZE]

    X_tensor = torch.tensor(X_subset, dtype=torch.float32)
    y_tensor = torch.tensor(y_subset, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TCNRegressor(
        input_channels=6,
        channel_sizes=[32, 64, 64],
        kernel_size=3,
        dropout=0.1,
        output_dim=3,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {DEVICE}")
    print(f"Subset size: {SUBSET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataset)

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.8f}")

    print("\nTraining complete.")

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(DEVICE)).cpu().numpy()

    print("\nFirst 5 predictions vs targets:")
    for i in range(5):
        print(f"Pred {i}: {preds[i]} | Target {i}: {y_subset[i]}")

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "subset_size": SUBSET_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "input_channels": 6,
                "output_dim": 3,
                "channel_sizes": [32, 64, 64],
                "kernel_size": 3,
                "dropout": 0.1,
                "seed": SEED,
            },
            "final_train_loss": epoch_loss,
        },
        checkpoint_dir / "tcn_subset_overfit.pt",
    )

    print(f"\nSaved subset checkpoint to: {checkpoint_dir / 'tcn_subset_overfit.pt'}")


if __name__ == "__main__":
    main()