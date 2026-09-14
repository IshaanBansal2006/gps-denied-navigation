"""Learn 15-state EKF noise parameters by backprop through batched GPS outages.

Parameters: IMU noise ``qc`` (4 groups: accel, gyro, accel-bias walk, gyro-bias
walk) and the LSTM velocity-measurement variance ``r_net`` (per axis).
Loss: mean squared velocity error over 30 s outages on train-sequence windows;
selection on MH_04 (val); report on MH_05 (test) against the hand-tuned EKF
(initialized at decision 019's best sigma_net) and the velocity-only filter
used by the project's headline pipeline.

    python3 scripts/learn_ekf_noise.py [--iters 40] [--windows-per-seq 8]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gps_denied_nav.batched_ekf import ekf_step, qc_diag  # noqa: E402
from gps_denied_nav.batched_ekf.function import ekf_step_cuda  # noqa: E402
from gps_denied_nav.batched_ekf.rollout import DT, OutageBatch, build_outage_batch, rollout  # noqa: E402
from gps_denied_nav.data.euroc import EUROC_TEST_SEQ, EUROC_TRAIN_SEQS, EUROC_VAL_SEQ, EuRoCSequence  # noqa: E402
from gps_denied_nav.filters.velocity_only import VelocityOnlyFilter  # noqa: E402
from gps_denied_nav.models import load_lstm_checkpoint  # noqa: E402

log = logging.getLogger("learn_ekf_noise")
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "lstm_v15_vel"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "lstm_v15.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "batched_ekf"
WARMUP = 2000
OUTAGE = 6000
STRIDE = 25
SIGMA_GPS = 0.02
SIGMA_NET_HAND = 0.01  # best strapdown sigma_tcn from the decision 019 sweep


@dataclass
class Split:
    name: str
    batch: OutageBatch
    seq_names: List[str]
    starts: List[int]


def lstm_velocity(seq: EuRoCSequence, device: torch.device) -> np.ndarray:
    """Causal LSTM v15 velocity for every sample (same as NavPipeline with no adapter)."""
    cache = CACHE_DIR / f"{seq.name}.npy"
    if cache.exists():
        return np.load(cache)
    model, norm = load_lstm_checkpoint(CHECKPOINT, device)
    x = (seq.imu - norm["x_mean"]) / norm["x_std"]
    with torch.no_grad():
        y, _ = model(torch.as_tensor(x[None], dtype=torch.float32, device=device))
    vel = y[0].cpu().numpy() * norm["y_std"] + norm["y_mean"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, vel)
    return vel


def random_starts(seq: EuRoCSequence, n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.integers(WARMUP, seq.n_samples - OUTAGE - 1, size=n).tolist())


def build_split(name: str, seq_names: List[str], n_per_seq: int, seed: int, device: torch.device) -> Split:
    batches, names, starts = [], [], []
    for sn in seq_names:
        seq = EuRoCSequence.load(sn, SEQUENCES_DIR)
        s = random_starts(seq, n_per_seq, seed)
        batches.append(build_outage_batch(seq, lstm_velocity(seq, device), s, WARMUP, OUTAGE, STRIDE, device=device))
        names += [sn] * len(s)
        starts += s
    cat = OutageBatch(
        x0=torch.cat([b.x0 for b in batches]),
        P0=torch.cat([b.P0 for b in batches]),
        imu=torch.cat([b.imu for b in batches], dim=1),
        z=torch.cat([b.z for b in batches], dim=1),
        mask=torch.cat([b.mask for b in batches], dim=1),
        gt_vel=torch.cat([b.gt_vel for b in batches], dim=1),
        warmup=WARMUP,
        outage=OUTAGE,
    )
    log.info("%s: %d windows from %s", name, cat.n, ", ".join(seq_names))
    return Split(name, cat, names, starts)


class NoiseParams(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        q = qc_diag()
        self.log_qc = torch.nn.Parameter(torch.log(torch.tensor(q[[0, 3, 6, 9]], dtype=torch.float64, device=device)))
        self.log_r_net = torch.nn.Parameter(torch.full((3,), 2 * np.log(SIGMA_NET_HAND), dtype=torch.float64, device=device))

    def qc(self) -> torch.Tensor:
        return torch.exp(self.log_qc).repeat_interleave(3)

    def r_net(self) -> torch.Tensor:
        return torch.exp(self.log_r_net)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "sigma_qc_groups[accel,gyro,ba_rw,bg_rw]": torch.exp(0.5 * self.log_qc).tolist(),
            "sigma_r_net[x,y,z]": torch.exp(0.5 * self.log_r_net).tolist(),
        }


def errors(vel: torch.Tensor, batch: OutageBatch) -> Dict[str, np.ndarray]:
    err = torch.linalg.norm(vel - batch.gt_vel[batch.warmup:], dim=-1)
    return {"final_vel": err[-1].cpu().numpy(), "mean_vel": err.mean(0).cpu().numpy()}


def evaluate_ekf(split: Split, qc: torch.Tensor, r_net: torch.Tensor, r_gps: torch.Tensor) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        vel = rollout(ekf_step_cuda, split.batch, qc, r_gps, r_net)
    return errors(vel, split.batch)


def evaluate_velocity_filter(split: Split) -> Dict[str, np.ndarray]:
    """Headline-pipeline filter (decision 019) on exactly the same windows and measurements."""
    b = split.batch
    z = b.z[b.warmup:].cpu().numpy()
    mask = b.mask[b.warmup:].cpu().numpy()
    gt = b.gt_vel[b.warmup:].cpu().numpy()
    v0 = b.gt_vel[b.warmup - 1].cpu().numpy()
    final, mean = [], []
    for j in range(b.n):
        f = VelocityOnlyFilter()
        f.reset(v0[j].astype(np.float64))
        err = np.zeros(b.outage)
        for t in range(b.outage):
            f.predict(DT)
            if mask[t, j] > 0.5:
                f.update(z[t, j])
            err[t] = np.linalg.norm(f.velocity - gt[t, j])
        final.append(err[-1])
        mean.append(err.mean())
    return {"final_vel": np.array(final), "mean_vel": np.array(mean)}


def summarize(e: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "final_vel_mean": float(e["final_vel"].mean()),
        "final_vel_p50": float(np.median(e["final_vel"])),
        "final_vel_p95": float(np.percentile(e["final_vel"], 95)),
        "mean_vel_mean": float(e["mean_vel"].mean()),
    }


def check_gradients(split: Split, params: NoiseParams, r_gps: torch.Tensor) -> None:
    """Kernel vs reference gradient on a short slice of real data before training."""
    b = split.batch
    short = OutageBatch(b.x0[:4], b.P0[:4], b.imu[:, :4], b.z[:, :4], b.mask[:, :4], b.gt_vel[:, :4], b.warmup, 200)
    grads = []
    for step in (ekf_step_cuda, ekf_step):
        params.zero_grad()
        vel = rollout(step, short, params.qc(), r_gps, params.r_net())
        ((vel - short.gt_vel[short.warmup:short.warmup + 200]) ** 2).mean().backward()
        grads.append(torch.cat([params.log_qc.grad.clone(), params.log_r_net.grad.clone()]))
    rel = ((grads[0] - grads[1]).abs().max() / grads[1].abs().max().clamp_min(1e-30)).item()
    log.info("real-data gradient check (kernel vs autograd reference): max rel diff %.2e", rel)
    if rel > 1e-6:
        raise RuntimeError(f"kernel gradient disagrees with reference on real data (rel {rel:.2e}); run pytest tests/test_batched_ekf_backward.py")
    params.zero_grad()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--windows-per-seq", type=int, default=8)
    parser.add_argument("--eval-windows", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this script trains through the custom CUDA EKF kernels.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    train = build_split("train", list(EUROC_TRAIN_SEQS), args.windows_per_seq, seed=0, device=device)
    val = build_split("val", [EUROC_VAL_SEQ], args.eval_windows, seed=42, device=device)
    test = build_split("test", [EUROC_TEST_SEQ], args.eval_windows, seed=42, device=device)

    params = NoiseParams(device)
    r_gps = torch.full((3,), SIGMA_GPS ** 2, dtype=torch.float64, device=device)
    check_gradients(train, params, r_gps)
    hand = {"qc": params.qc().detach().clone(), "r_net": params.r_net().detach().clone()}

    opt = torch.optim.Adam(params.parameters(), lr=args.lr)
    history = []
    best = {"val_mean_vel": float("inf"), "iter": -1, "state": None}
    for it in range(args.iters + 1):
        if it % args.eval_every == 0 or it == args.iters:
            val_m = summarize(evaluate_ekf(val, params.qc().detach(), params.r_net().detach(), r_gps))
            history.append({"iter": it, "val": val_m, **params.as_dict()})
            log.info("iter %3d  val mean_vel %.4f  final_vel %.4f  %s", it, val_m["mean_vel_mean"], val_m["final_vel_mean"], params.as_dict())
            if val_m["mean_vel_mean"] < best["val_mean_vel"]:
                best = {"val_mean_vel": val_m["mean_vel_mean"], "iter": it,
                        "state": {k: v.detach().clone() for k, v in params.state_dict().items()}}
        if it == args.iters:
            break
        t0 = time.perf_counter()
        opt.zero_grad()
        vel = rollout(ekf_step_cuda, train.batch, params.qc(), r_gps, params.r_net())
        loss = ((vel - train.batch.gt_vel[WARMUP:]) ** 2).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), 1.0)
        opt.step()
        history[-1].setdefault("train_loss", []).append(float(loss))
        log.info("iter %3d  train loss %.4f  (%.1fs)", it, float(loss), time.perf_counter() - t0)

    params.load_state_dict(best["state"])
    report = {
        "protocol": {"warmup_samples": WARMUP, "outage_samples": OUTAGE, "stride": STRIDE, "sigma_gps": SIGMA_GPS,
                     "train_windows": train.batch.n, "eval_windows": args.eval_windows, "val_seq": EUROC_VAL_SEQ,
                     "test_seq": EUROC_TEST_SEQ, "test_starts": test.starts, "selected_iter": best["iter"]},
        "learned_params": params.as_dict(),
        "history": history,
    }
    for split in (val, test):
        report[split.name] = {
            "velocity_only_filter": summarize(evaluate_velocity_filter(split)),
            "ekf_hand_tuned": summarize(evaluate_ekf(split, hand["qc"], hand["r_net"], r_gps)),
            "ekf_learned": summarize(evaluate_ekf(split, params.qc().detach(), params.r_net().detach(), r_gps)),
        }
        for system, m in report[split.name].items():
            log.info("%-4s %-22s final_vel mean %.4f p50 %.4f p95 %.4f | mean_vel %.4f", split.name, system,
                     m["final_vel_mean"], m["final_vel_p50"], m["final_vel_p95"], m["mean_vel_mean"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "learned_noise.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
