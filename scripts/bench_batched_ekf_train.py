"""Training-step benchmark: forward + backward through a T-step batched EKF rollout.

Compares autograd through the PyTorch reference against the custom
``EKFStepFunction`` (hand-written CUDA backward). Reports rollout-steps/s
(N filters × T steps per second) and peak GPU memory.

    python3 scripts/bench_batched_ekf_train.py [--dtype float32] [--steps 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gps_denied_nav.batched_ekf import ekf_step  # noqa: E402
from gps_denied_nav.batched_ekf.function import ekf_step_cuda  # noqa: E402
from tests._ekf_scenarios import DT, make_scenario, torch_inputs  # noqa: E402

log = logging.getLogger("bench_batched_ekf_train")
RESULTS_DIR = PROJECT_ROOT / "results" / "batched_ekf"


@dataclass
class Row:
    backend: str
    dtype: str
    n_filters: int
    steps: int
    seconds: float
    filter_steps_per_s: float
    peak_mem_mb: float


def train_step(step: Callable, inputs: tuple, steps: int) -> None:
    x, P, imu, z, r, qc, mask = inputs
    z_leaf = z.clone().requires_grad_(True)
    r_leaf = r.clone().requires_grad_(True)
    loss = x.new_zeros(())
    for t in range(steps):
        x, P = step(x, P, imu[t], z_leaf[t], r_leaf[t], qc, mask[t], DT)
        loss = loss + (x[:, 3:6] ** 2).sum()
    loss.backward()


def measure(step: Callable, n: int, steps: int, dtype: torch.dtype) -> tuple:
    sc = make_scenario(n=n, steps=steps, seed=0)
    inputs = torch_inputs(sc, dtype, "cuda")
    train_step(step, inputs, 2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    train_step(step, inputs, steps)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return seconds, (torch.cuda.max_memory_allocated() - base) / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 1024, 8192])
    parser.add_argument("--tag", default="train")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", default=["torch_ref_autograd", "cuda_kernel_fn"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dtype = getattr(torch, args.dtype)

    rows: List[Row] = []
    for n in args.sizes:
        for name, fn in (("torch_ref_autograd", ekf_step), ("cuda_kernel_fn", ekf_step_cuda)):
            if name not in args.backends:
                continue
            try:
                runs = [measure(fn, n, args.steps, dtype) for _ in range(args.repeats)]
                seconds = statistics.median(r[0] for r in runs)
                mem = max(r[1] for r in runs)
            except torch.cuda.OutOfMemoryError:
                log.warning("%s N=%d OOM at %d steps; skipping", name, n, args.steps)
                torch.cuda.empty_cache()
                continue
            row = Row(name, args.dtype, n, args.steps, seconds, n * args.steps / seconds, mem)
            rows.append(row)
            log.info("%-20s N=%-6d %10.0f filter-steps/s  peak %8.1f MB", name, n, row.filter_steps_per_s, mem)
            torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bench_{args.tag}_{args.dtype}.json"
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    out.write_text(json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
