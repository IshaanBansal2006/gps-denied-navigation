"""
LSTM v14: end-to-end navigation loss, 10s simulated GPS outage.

Instead of per-step velocity MSE, simulate GPS outages during training and
backprop through the differentiable velocity-only Kalman filter to minimize
actual navigation drift.

Architecture:
  - Same LSTMRegressor as v12/v13 (hidden=128, 2 layers, dropout=0.3)
  - Reuses v13 normalization stats (x_mean/std, y_mean/std)
  - Differentiable vel-only Kalman filter for training loss

Training loop:
  1. Sample random sequence + random outage position
  2. Warmup LSTM on WARMUP_LEN=1000 pre-outage samples (no_grad)
  3. Detach hidden state — gradient starts at outage boundary
  4. Run differentiable filter for OUTAGE_LEN=2000 (10s) steps
  5. Loss = mean(||v_filter - v_gt||) over outage (physical units, m/s)
  6. Accumulate 4 outages before optimizer.step()

Val metric: mean velocity error at 30s on MH_04_difficult (single fixed outage)
Early stopping on val metric (patience=20).

Saves:
  checkpoints/lstm_v14.pt
  results/lstm_v14/{test_metrics,normalization_stats,loss_history}.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "lstm_v14"

TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V1_03_difficult"]
VAL_SEQ = "MH_04_difficult"
TEST_SEQ = "MH_05_difficult"

IMU_COLS = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
VEL_COLS = ["gt_vel_x", "gt_vel_y", "gt_vel_z"]

# Training config
OUTAGE_LEN = 2000       # 10s @ 200 Hz
WARMUP_LEN = 1000       # 5s pre-outage warmup (no_grad)
ACCUMULATE = 4          # gradient accumulation steps per optimizer.step()
STRIDE = 25             # Kalman update every 25 IMU samples (8 Hz)
SIGMA_PROCESS = 0.5
R_DIAG = [0.61**2, 0.67**2, 0.25**2]   # per-axis measurement noise

# Training schedule
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 20
STEPS_PER_EPOCH = 200   # gradient steps per epoch (each = ACCUMULATE outage sims)

# Val: fixed 30s outage on MH_04_difficult
VAL_OUTAGE_LEN = 6000   # 30s


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 output_dim=3, dropout=0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.head(self.dropout(out)), state


def load_norm_stats() -> Dict:
    """Load v13 normalization stats — reuse for direct comparability."""
    p = PROJECT_ROOT / "results" / "lstm_v13" / "normalization_stats.json"
    with open(p) as f:
        ns = json.load(f)
    return {
        "x_mean": np.array(ns["x_mean"], dtype=np.float32),
        "x_std":  np.array(ns["x_std"],  dtype=np.float32),
        "y_mean": np.array(ns["y_mean"], dtype=np.float32),
        "y_std":  np.array(ns["y_std"],  dtype=np.float32),
    }


def load_seq_raw(seq_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    csv_path = SEQUENCES_DIR / seq_name / "imu_aligned.csv"
    if not csv_path.exists():
        print(f"[skip] {seq_name}: not found")
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in IMU_COLS + VEL_COLS if c not in df.columns]
    if missing:
        print(f"[skip] {seq_name}: missing {missing}")
        return None
    return df[IMU_COLS].to_numpy(dtype=np.float32), df[VEL_COLS].to_numpy(dtype=np.float32)


def normalize_seq(imu: np.ndarray, vel: np.ndarray, ns: Dict) -> Tuple[np.ndarray, np.ndarray]:
    imu_n = (imu - ns["x_mean"]) / ns["x_std"]
    vel_n = (vel - ns["y_mean"]) / ns["y_std"]
    return imu_n, vel_n


def run_differentiable_filter(
    model: LSTMRegressor,
    imu_tensor: torch.Tensor,
    vel_init: torch.Tensor,
    lstm_state: Tuple,
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable velocity-only Kalman filter for training.

    imu_tensor:  (T, 6) normalized IMU on device — no grad
    vel_init:    (3,)   initial filter velocity in physical units — detached
    lstm_state:  detached (h, c) from warmup phase
    Returns:     (T+1, 3) filter velocity in physical units, with gradient
    """
    T = imu_tensor.shape[0]

    v = vel_init.detach().clone().to(device)
    P = torch.eye(3, device=device) * 0.5
    R_tcn = torch.diag(torch.tensor(R_DIAG, dtype=torch.float32, device=device))
    Q_rate = SIGMA_PROCESS ** 2 * torch.eye(3, device=device)
    I3 = torch.eye(3, device=device)
    dt = 0.005

    state = lstm_state
    steps_since_update = 0
    vel_history: List[torch.Tensor] = [v.unsqueeze(0)]

    for t in range(T):
        # Predict: grow covariance
        P = P + Q_rate * dt

        # LSTM step (gradient flows)
        x_t = imu_tensor[t:t + 1].unsqueeze(0)   # (1, 1, 6)
        y_norm, state = model(x_t, state)
        v_pred_norm = y_norm[0, 0]                # (3,) normalized
        v_pred = v_pred_norm * y_std_t + y_mean_t  # (3,) physical units

        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            S = P + R_tcn
            K = P @ torch.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (I3 - K) @ P

        vel_history.append(v.unsqueeze(0))

    return torch.cat(vel_history, dim=0)  # (T+1, 3)


def warmup_lstm(
    model: LSTMRegressor,
    imu_norm: np.ndarray,
    outage_start: int,
    device: torch.device,
) -> Tuple:
    """Run LSTM over pre-outage IMU, return detached hidden state."""
    warmup_start = max(0, outage_start - WARMUP_LEN)
    with torch.no_grad():
        state = None
        for k in range(warmup_start, outage_start):
            x = torch.tensor(imu_norm[k:k + 1][None], dtype=torch.float32, device=device)
            _, state = model(x, state)
    if state is None:
        return None
    h = state[0].detach()
    c = state[1].detach()
    return (h, c)


def train_step(
    model: LSTMRegressor,
    seqs_norm: List[Tuple[np.ndarray, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    rng: np.random.Generator,
) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for _ in range(ACCUMULATE):
        seq_idx = rng.integers(0, len(seqs_norm))
        imu_norm, vel_norm = seqs_norm[seq_idx]
        T = len(imu_norm)
        min_start = WARMUP_LEN
        max_start = T - OUTAGE_LEN - 1
        if max_start <= min_start:
            continue
        outage_start = int(rng.integers(min_start, max_start))

        # Warmup (no grad)
        state = warmup_lstm(model, imu_norm, outage_start, device)

        # Outage tensors (no requires_grad — inputs only)
        outage_imu = torch.tensor(
            imu_norm[outage_start:outage_start + OUTAGE_LEN],
            dtype=torch.float32, device=device)
        outage_vel_norm = torch.tensor(
            vel_norm[outage_start:outage_start + OUTAGE_LEN],
            dtype=torch.float32, device=device)

        # Initial filter velocity from gt at outage start
        vel_init = outage_vel_norm[0] * y_std_t + y_mean_t  # physical units

        # Run differentiable filter
        vel_filter = run_differentiable_filter(
            model, outage_imu, vel_init, state, device, y_mean_t, y_std_t)
        # vel_filter: (OUTAGE_LEN+1, 3) physical

        # Ground truth in physical units
        gt_denorm = outage_vel_norm * y_std_t + y_mean_t  # (OUTAGE_LEN, 3)

        # Mean velocity error over outage
        errors = (vel_filter[1:] - gt_denorm).norm(dim=-1)  # (OUTAGE_LEN,)
        loss = errors.mean() / ACCUMULATE
        loss.backward()
        total_loss += loss.item()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss


@torch.no_grad()
def validate(
    model: LSTMRegressor,
    val_norm: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
) -> Tuple[float, float]:
    """Fixed 30s outage at 40% into MH_04_difficult. Returns (mean_err, final_err)."""
    model.eval()
    imu_norm, vel_norm = val_norm
    T = len(imu_norm)
    outage_start = int(T * 0.4)
    outage_len = min(VAL_OUTAGE_LEN, T - outage_start - 1)

    # Full pre-outage warmup
    state = None
    for k in range(outage_start):
        x = torch.tensor(imu_norm[k:k + 1][None], dtype=torch.float32, device=device)
        _, state = model(x, state)

    v = (torch.tensor(vel_norm[outage_start], dtype=torch.float32, device=device)
         * y_std_t + y_mean_t)
    P = torch.eye(3, device=device) * 0.5
    R_tcn = torch.diag(torch.tensor(R_DIAG, dtype=torch.float32, device=device))
    Q_rate = SIGMA_PROCESS ** 2 * torch.eye(3, device=device)
    I3 = torch.eye(3, device=device)
    dt = 0.005

    gt = (torch.tensor(vel_norm[outage_start:outage_start + outage_len],
                        dtype=torch.float32, device=device)
          * y_std_t + y_mean_t)

    errors = []
    steps_since_update = 0
    for t in range(outage_len):
        P = P + Q_rate * dt
        x = torch.tensor(imu_norm[outage_start + t:outage_start + t + 1][None],
                          dtype=torch.float32, device=device)
        y_norm_t, state = model(x, state)
        v_pred = y_norm_t[0, 0] * y_std_t + y_mean_t

        steps_since_update += 1
        if steps_since_update >= STRIDE:
            steps_since_update = 0
            S = P + R_tcn
            K = P @ torch.linalg.inv(S)
            v = v + K @ (v_pred - v)
            P = (I3 - K) @ P

        errors.append((v - gt[t]).norm().item())

    return float(np.mean(errors)), float(errors[-1])


@torch.no_grad()
def compute_chunk_metrics(
    model: LSTMRegressor,
    imu_norm: np.ndarray,
    vel_norm: np.ndarray,
    device: torch.device,
) -> Dict:
    """Chunk-based R²/corr metrics on test set (for comparison with v12/v13)."""
    model.eval()
    CHUNK_LEN = 400
    CHUNK_STRIDE = 200
    T = len(imu_norm)
    preds, targets = [], []
    for start in range(0, T - CHUNK_LEN + 1, CHUNK_STRIDE):
        xb = torch.tensor(imu_norm[start:start + CHUNK_LEN][None],
                           dtype=torch.float32, device=device)
        yb = vel_norm[start:start + CHUNK_LEN]
        pred, _ = model(xb)
        preds.append(pred[0, -1, :].cpu().numpy())
        targets.append(yb[-1])

    p = np.stack(preds)
    t = np.stack(targets)
    diff = p - t
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    axis_mse = [float(np.mean(diff[:, i] ** 2)) for i in range(3)]
    t_var = np.var(t, axis=0)
    axis_r2 = [float(1 - axis_mse[i] / t_var[i]) if t_var[i] > 1e-8 else float("nan")
               for i in range(3)]

    def pearson(a, b):
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        return float((ac * bc).sum() / d) if d > 1e-8 else 0.0

    corr = [pearson(p[:, i], t[:, i]) for i in range(3)]
    return {
        "mse": mse, "mae": mae,
        "r2_mean": float(np.nanmean(axis_r2)),
        "r2_x": axis_r2[0], "r2_y": axis_r2[1], "r2_z": axis_r2[2],
        "corr_x": corr[0], "corr_y": corr[1], "corr_z": corr[2],
        "mse_x": axis_mse[0], "mse_y": axis_mse[1], "mse_z": axis_mse[2],
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    ns = load_norm_stats()
    y_mean_t = torch.tensor(ns["y_mean"], dtype=torch.float32, device=device)
    y_std_t  = torch.tensor(ns["y_std"],  dtype=torch.float32, device=device)

    print("Loading training sequences...")
    seqs_norm: List[Tuple[np.ndarray, np.ndarray]] = []
    for seq in TRAIN_SEQS:
        raw = load_seq_raw(seq)
        if raw is None:
            continue
        imu, vel = raw
        imu_n, vel_n = normalize_seq(imu, vel, ns)
        seqs_norm.append((imu_n, vel_n))
        print(f"  {seq}: {len(imu_n)} samples")

    print(f"Loading val: {VAL_SEQ}")
    val_raw = load_seq_raw(VAL_SEQ)
    assert val_raw is not None
    val_norm = normalize_seq(val_raw[0], val_raw[1], ns)

    print(f"Loading test: {TEST_SEQ}")
    test_raw = load_seq_raw(TEST_SEQ)
    assert test_raw is not None
    test_norm = normalize_seq(test_raw[0], test_raw[1], ns)

    model = LSTMRegressor(input_size=6, hidden_size=128, num_layers=2,
                           output_dim=3, dropout=0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    rng = np.random.default_rng(42)
    best_val, no_improve = math.inf, 0
    best_ckpt = CHECKPOINT_DIR / "lstm_v14.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    norm_stats_out = {k: v.tolist() for k, v in ns.items()}
    save_json(RESULTS_DIR / "normalization_stats.json", norm_stats_out)

    cfg = {
        "model": "lstm", "hidden_size": 128, "num_layers": 2, "dropout": 0.3,
        "loss": "nav_loss_mean_velocity_error",
        "outage_len_s": OUTAGE_LEN / 200, "warmup_len_s": WARMUP_LEN / 200,
        "accumulate": ACCUMULATE, "stride": STRIDE, "sigma_process": SIGMA_PROCESS,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "steps_per_epoch": STEPS_PER_EPOCH,
        "train_seqs": TRAIN_SEQS,
    }

    history: Dict[str, List] = {"train_loss": [], "val_mean_err": [], "val_final_err": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        epoch_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            epoch_loss += train_step(model, seqs_norm, optimizer, device,
                                     y_mean_t, y_std_t, rng)
        train_loss = epoch_loss / STEPS_PER_EPOCH

        # Val
        val_mean, val_final = validate(model, val_norm, device, y_mean_t, y_std_t)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_mean_err"].append(val_mean)
        history["val_final_err"].append(val_final)

        improved = val_mean < best_val
        if improved:
            best_val, no_improve = val_mean, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mean_err": best_val,
                "config": cfg,
                "norm_stats": norm_stats_out,
            }, best_ckpt)
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val_mean {val_mean:.4f} | val_final {val_final:.4f}"
              + (" | best" if improved else ""))

        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch}.")
            break

    save_json(RESULTS_DIR / "loss_history.json", history)

    # Load best and compute test metrics
    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    chunk_metrics = compute_chunk_metrics(model, test_norm[0], test_norm[1], device)
    test_val_mean, test_val_final = validate(model, test_norm, device, y_mean_t, y_std_t)

    test_metrics = {
        **chunk_metrics,
        "nav_val_mean_err": test_val_mean,
        "nav_val_final_err": test_val_final,
        "best_val_mean_err": best["best_val_mean_err"],
        "best_epoch": best["epoch"],
    }
    save_json(RESULTS_DIR / "test_metrics.json", test_metrics)

    print(f"\nBest epoch: {best['epoch']}  Val mean err: {best['best_val_mean_err']:.4f} m/s")
    print("Test metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    for ver, path in [("v13", "results/lstm_v13/test_metrics.json"),
                      ("v12", "results/lstm_v12/test_metrics.json")]:
        p = PROJECT_ROOT / path
        if p.exists():
            with open(p) as f:
                ref = json.load(f)
            print(f"\n--- vs {ver} ---")
            for k in ["r2_mean", "corr_x", "corr_y", "corr_z"]:
                if k in ref:
                    delta = chunk_metrics[k] - ref[k]
                    print(f"  {k}: {chunk_metrics[k]:.4f} ({ver}: {ref[k]:.4f}, "
                          f"{'+' if delta >= 0 else ''}{delta:.4f})")


if __name__ == "__main__":
    main()
