#!/usr/bin/env python3
"""Assemble multi-sequence dataset from processed sequences.

Run from repo root after process_sequence.py has been run for each sequence.
Reads split config, concatenates windows per split, normalises (train stats only).

Usage:
    python3 scripts/build_dataset.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SPLITS_DIR = Path("data/splits")
SEQUENCES_DIR = Path("data/sequences")

# Bag locations on this machine
BAG_ROOTS = [
    Path("/home/ishaan/machine_hall/machine_hall"),
    Path("/home/ishaan/vicon_room1/vicon_room1"),
    Path("/home/ishaan/datasets/euroc"),
    Path("data"),
]

SPLIT_CONFIG: dict[str, list[str]] = {
    "train": ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V1_03_difficult"],
    "val":   ["MH_04_difficult"],
    "test":  ["MH_05_difficult"],
}

WINDOW_SIZE = 200
STRIDE = 25


def find_bag(seq_name: str) -> Path | None:
    for root in BAG_ROOTS:
        candidate = root / seq_name / f"{seq_name}.bag"
        if candidate.exists():
            return candidate
        candidate = root / f"{seq_name}.bag"
        if candidate.exists():
            return candidate
    return None


def ensure_processed(seq_name: str) -> bool:
    x_path = SEQUENCES_DIR / seq_name / "X_windows.npy"
    if x_path.exists():
        return True

    bag = find_bag(seq_name)
    if bag is None:
        log.warning(f"[skip] Bag not found for {seq_name} — skipping.")
        return False

    log.info(f"[process] Running process_sequence.py for {seq_name} ...")
    result = subprocess.run(
        [sys.executable, "scripts/process_sequence.py", str(bag), seq_name],
        check=False,
    )
    if result.returncode != 0:
        log.warning(f"[skip] process_sequence.py failed for {seq_name} — skipping.")
        return False
    return True


def load_windows(seq_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    x_path = SEQUENCES_DIR / seq_name / "X_windows.npy"
    y_path = SEQUENCES_DIR / seq_name / "y_delta_v.npy"
    if not x_path.exists():
        return None
    return np.load(x_path), np.load(y_path)


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    split_arrays: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        split: ([], []) for split in SPLIT_CONFIG
    }

    for split, sequences in SPLIT_CONFIG.items():
        for seq_name in sequences:
            if not ensure_processed(seq_name):
                continue
            result = load_windows(seq_name)
            if result is None:
                log.warning(f"[skip] No windows for {seq_name}")
                continue
            X, y = result
            split_arrays[split][0].append(X)
            split_arrays[split][1].append(y)
            log.info(f"[loaded] {seq_name}: {len(X)} windows → {split}")

    assembled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, (xs, ys) in split_arrays.items():
        if not xs:
            raise RuntimeError(f"No data for split '{split}'. Check bag paths.")
        assembled[split] = (
            np.concatenate(xs, axis=0).astype(np.float32),
            np.concatenate(ys, axis=0).astype(np.float32),
        )

    X_train, y_train = assembled["train"]
    X_val, y_val = assembled["val"]
    X_test, y_test = assembled["test"]

    # Normalise using train stats only
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val", X_val), ("y_val", y_val),
        ("X_test", X_test), ("y_test", y_test),
    ]:
        np.save(SPLITS_DIR / f"{name}.npy", arr)

    stats = {
        "split_config": SPLIT_CONFIG,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "feature_order": ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"],
        "feature_mean": feat_mean.reshape(-1).tolist(),
        "feature_std": feat_std.reshape(-1).tolist(),
    }
    with open(SPLITS_DIR / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info("\n=== Dataset assembled ===")
    log.info(f"Train: {len(X_train)} windows from {SPLIT_CONFIG['train']}")
    log.info(f"Val:   {len(X_val)} windows from {SPLIT_CONFIG['val']}")
    log.info(f"Test:  {len(X_test)} windows from {SPLIT_CONFIG['test']}")
    log.info(f"Saved to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
