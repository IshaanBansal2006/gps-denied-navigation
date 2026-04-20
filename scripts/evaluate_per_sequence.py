#!/usr/bin/env python3
"""Evaluate the trained TCN on each sequence individually.

Shows how well the model generalises across easy / medium / hard flights.
Sequences in the training set will score better — that gap reveals overfitting.

Usage:
    python3 scripts/evaluate_per_sequence.py [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.tcn import TCNRegressor

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "per_sequence_eval"

WINDOW_SIZE = 200
STRIDE = 25
IMU_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
GT_VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]

SPLIT_MEMBERSHIP = {
    "MH_01_easy": "train",
    "MH_02_easy": "train",
    "MH_03_medium": "train",
    "V1_01_easy": "train",
    "V1_02_medium": "train",
    "MH_04_difficult": "val",
    "MH_05_difficult": "test",
}


def load_sequence_windows(seq_name: str, norm_stats: dict) -> tuple[np.ndarray, np.ndarray] | None:
    x_path = SEQUENCES_DIR / seq_name / "X_windows.npy"
    y_path = SEQUENCES_DIR / seq_name / "y_delta_v.npy"
    if not x_path.exists():
        return None
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    mean = np.array(norm_stats["feature_mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.array(norm_stats["feature_std"], dtype=np.float32).reshape(1, 1, -1)
    X = (X - mean) / std
    return X, y


def evaluate(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=256, shuffle=False,
    )
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            targets.append(yb.numpy())

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    diff = p - t

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    zero_mse = float(np.mean(t ** 2))
    axis_mse = np.mean(diff ** 2, axis=0).tolist()

    t_var = t.var(axis=0)
    axis_r2 = [float(1 - axis_mse[i] / t_var[i]) if t_var[i] > 1e-8 else float("nan") for i in range(3)]
    r2_mean = float(np.nanmean(axis_r2))

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        return float(ac @ bc / d) if d > 1e-8 else 0.0

    corr = [pearson(p[:, i], t[:, i]) for i in range(3)]

    return {
        "n_windows": len(p),
        "mse": mse, "mae": mae, "r2_mean": r2_mean,
        "zero_predictor_mse": zero_mse,
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "tcn_multi.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    model = TCNRegressor(
        input_channels=cfg.get("input_channels", 6),
        output_dim=cfg.get("output_dim", 3),
        channel_sizes=cfg.get("channel_sizes", [32, 64, 64]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    with open(SPLITS_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print(f"\n{'Sequence':<22} {'Split':<8} {'Windows':>7} {'MSE':>8} {'ZeroMSE':>8} {'R²':>7} {'corr_x':>7} {'corr_y':>7} {'corr_z':>7}")
    print("-" * 90)

    for seq_name, split in sorted(SPLIT_MEMBERSHIP.items(), key=lambda x: x[1]):
        result = load_sequence_windows(seq_name, norm_stats)
        if result is None:
            print(f"{seq_name:<22} {split:<8} {'MISSING':>7}")
            continue
        X, y = result
        m = evaluate(model, X, y, device)
        all_results[seq_name] = {"split": split, **m}

        print(
            f"{seq_name:<22} {split:<8} {m['n_windows']:>7} "
            f"{m['mse']:>8.4f} {m['zero_predictor_mse']:>8.4f} "
            f"{m['r2_mean']:>7.3f} "
            f"{m['corr_x']:>7.3f} {m['corr_y']:>7.3f} {m['corr_z']:>7.3f}"
        )

    out_path = RESULTS_DIR / "all_sequences.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
