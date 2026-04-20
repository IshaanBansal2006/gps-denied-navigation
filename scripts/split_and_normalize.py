#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np


X_INPUT = Path("data/processed/X_windows.npy")
Y_INPUT = Path("data/processed/y_delta_v.npy")

OUTPUT_DIR = Path("data/processed/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main() -> None:
    if not X_INPUT.exists():
        raise FileNotFoundError(f"Missing file: {X_INPUT}")
    if not Y_INPUT.exists():
        raise FileNotFoundError(f"Missing file: {Y_INPUT}")

    X = np.load(X_INPUT)
    y = np.load(Y_INPUT)

    n = len(X)
    if len(y) != n:
        raise ValueError("X and y do not have the same number of samples.")

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_end = n_train
    val_end = n_train + n_val

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    # Compute normalization stats from TRAIN ONLY.
    # X shape: (num_samples, window_size, num_features)
    # We want one mean/std per feature channel across all train samples and time steps.
    feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train.std(axis=(0, 1), keepdims=True)

    # Prevent divide-by-zero in case a feature is constant
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_array(OUTPUT_DIR / "X_train.npy", X_train_norm.astype(np.float32))
    save_array(OUTPUT_DIR / "y_train.npy", y_train.astype(np.float32))

    save_array(OUTPUT_DIR / "X_val.npy", X_val_norm.astype(np.float32))
    save_array(OUTPUT_DIR / "y_val.npy", y_val.astype(np.float32))

    save_array(OUTPUT_DIR / "X_test.npy", X_test_norm.astype(np.float32))
    save_array(OUTPUT_DIR / "y_test.npy", y_test.astype(np.float32))

    # Save raw train stats for reuse later during inference / deployment
    stats = {
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "num_samples_total": int(n),
        "num_samples_train": int(len(X_train)),
        "num_samples_val": int(len(X_val)),
        "num_samples_test": int(len(X_test)),
        "feature_order": [
            "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z",
        ],
        "feature_mean": feature_mean.reshape(-1).tolist(),
        "feature_std": feature_std.reshape(-1).tolist(),
    }

    with open(OUTPUT_DIR / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Chronological split and normalization complete.")
    print(f"Total samples: {n}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Test samples:  {len(X_test)}")
    print()
    print("Saved files:")
    print(f"  {OUTPUT_DIR / 'X_train.npy'}")
    print(f"  {OUTPUT_DIR / 'y_train.npy'}")
    print(f"  {OUTPUT_DIR / 'X_val.npy'}")
    print(f"  {OUTPUT_DIR / 'y_val.npy'}")
    print(f"  {OUTPUT_DIR / 'X_test.npy'}")
    print(f"  {OUTPUT_DIR / 'y_test.npy'}")
    print(f"  {OUTPUT_DIR / 'normalization_stats.json'}")
    print()
    print("Feature mean:", feature_mean.reshape(-1))
    print("Feature std: ", feature_std.reshape(-1))


if __name__ == "__main__":
    main()