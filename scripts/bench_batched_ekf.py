"""Throughput benchmark for the batched EKF step (predict + velocity update).

Backends: NumPy EKF15 loop, PyTorch reference (CPU/GPU), C++ extension on CPU
threads, custom CUDA kernel. Reports filter-steps per second.

    python3 scripts/bench_batched_ekf.py [--dtype float32|float64] [--steps 20]
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

from gps_denied_nav.batched_ekf import _ext, ekf_step  # noqa: E402
from tests._ekf_scenarios import DT, make_scenario, run_numpy, torch_inputs  # noqa: E402

log = logging.getLogger("bench_batched_ekf")
RESULTS_DIR = PROJECT_ROOT / "results" / "batched_ekf"


@dataclass
class Row:
    backend: str
    dtype: str
    n_filters: int
    steps: int
    seconds: float
    filter_steps_per_s: float


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_torch(step: Callable, n: int, steps: int, dtype: torch.dtype, device: str) -> float:
    sc = make_scenario(n=n, steps=steps, seed=0)
    x, P, imu, z, r, qc, mask = torch_inputs(sc, dtype, device)
    frames = [(imu[t].contiguous(), z[t].contiguous(), r[t].contiguous(), mask[t].contiguous()) for t in range(steps)]
    for _ in range(3):
        step(x, P, *frames[0][:3], qc, frames[0][3], DT)
    _sync(device)
    t0 = time.perf_counter()
    for imu_t, z_t, r_t, m_t in frames:
        x, P = step(x, P, imu_t, z_t, r_t, qc, m_t, DT)
    _sync(device)
    return time.perf_counter() - t0


def time_numpy(n: int, steps: int) -> float:
    sc = make_scenario(n=n, steps=steps, seed=0)
    t0 = time.perf_counter()
    run_numpy(sc)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 64, 1024, 16384, 65536])
    parser.add_argument("--tag", default="forward")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backends", nargs="+", default=None,
                        help="subset of: numpy_ekf15 torch_ref_cpu torch_ref_cuda cpp_cpu_threads cuda_kernel")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dtype = getattr(torch, args.dtype)
    ext = _ext.load()
    backends = {
        "torch_ref_cpu": (ekf_step, "cpu"),
        "torch_ref_cuda": (ekf_step, "cuda"),
        "cpp_cpu_threads": (ext.step_forward, "cpu"),
        "cuda_kernel": (ext.step_forward, "cuda"),
    }

    rows: List[Row] = []

    def record(backend: str, n: int, seconds: float, steps: int) -> None:
        row = Row(backend, args.dtype, n, steps, seconds, n * steps / seconds)
        rows.append(row)
        log.info("%-16s N=%-6d %10.0f filter-steps/s", backend, n, row.filter_steps_per_s)

    wanted = set(args.backends or ["numpy_ekf15", *backends])
    if "numpy_ekf15" in wanted:
        numpy_n = 16
        record("numpy_ekf15", numpy_n, time_numpy(numpy_n, args.steps), args.steps)
    for n in args.sizes:
        for name, (fn, device) in backends.items():
            if name not in wanted or (name == "torch_ref_cpu" and n > 16384):
                continue
            times = [time_torch(fn, n, args.steps, dtype, device) for _ in range(args.repeats)]
            record(name, n, statistics.median(times), args.steps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bench_{args.tag}_{args.dtype}.json"
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
    }
    out.write_text(json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
