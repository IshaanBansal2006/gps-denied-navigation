"""Monte Carlo outage eval on MH_05 — 20 random outage start positions.

Turns the single-point headline (0.259 m/s at start_frac=0.4) into a
distribution: mean / std / p50 / p95 / max across N=20 random outage start
positions drawn uniformly from the legal range (≥ WARMUP_LEN samples
before, ≥ OUTAGE_LEN + 1 samples after).

Runs all three systems for each random window:
  - vanilla v15 + filter
  - v15 + filter + RLS
  - v15 + filter + continuous adapter

Output:
  results/monte_carlo/test_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gps_denied_nav import EuRoCSequence, NavPipeline, OutageEvaluator
from gps_denied_nav.adaptation import ContinuousAdapter, RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter
from gps_denied_nav.models import load_lstm_checkpoint

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
RESULTS_DIR = PROJECT_ROOT / "results" / "monte_carlo"

SEQ_NAME = "MH_05_difficult"
OUTAGE_S = 30.0
N_WINDOWS = 20
WARMUP_LEN = 2000  # samples ≈ 10 s pre-outage warmup needed
OUTAGE_LEN = 6000  # samples ≈ 30 s
SEED = 42


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def make_pipeline(kind: str, model, norm, device):
    if kind == "vanilla":
        return NavPipeline(model=model, adapter=None,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25), None
    rls = RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
    if kind == "rls":
        return NavPipeline(model=model, adapter=rls,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25), None
    if kind == "continuous":
        cont = ContinuousAdapter(rls=rls, alpha_smooth=0.0,
                                  ema_alpha=0.95, outage_lambda=1.0)
        return NavPipeline(model=model, adapter=rls,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25,
                            continuous_adapter=cont), cont
    raise ValueError(kind)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(SEQ_NAME, SEQUENCES_DIR)
    model, norm = load_lstm_checkpoint(PROJECT_ROOT / "checkpoints" / "lstm_v15.pt",
                                        device)

    # Sample N_WINDOWS random outage starts.
    rng = np.random.default_rng(SEED)
    min_start = WARMUP_LEN
    max_start = seq.n_samples - OUTAGE_LEN - 1
    starts = sorted(rng.integers(min_start, max_start, size=N_WINDOWS).tolist())
    fracs = [float(s) / seq.n_samples for s in starts]

    print(f"Sequence: {SEQ_NAME}  N={N_WINDOWS}  outage={OUTAGE_S}s  seed={SEED}")
    print(f"Start positions (frac): {[f'{f:.3f}' for f in fracs[:5]]}... ({len(fracs)} total)\n")

    per_system: dict = {k: {"final_vel": [], "mean_vel": [],
                             "final_pos": [], "mean_pos": []}
                         for k in ("vanilla", "rls", "continuous")}

    for i, (s, f) in enumerate(zip(starts, fracs)):
        ev = OutageEvaluator(seq, outage_start_frac=f)
        print(f"[{i + 1:02d}/{N_WINDOWS}]  start={s}  frac={f:.3f}")
        for kind in ("vanilla", "rls", "continuous"):
            pipeline, _ = make_pipeline(kind, model, norm, device)
            _, m = ev.evaluate(pipeline, outage_duration_s=OUTAGE_S)
            per_system[kind]["final_vel"].append(m.final_velocity_error)
            per_system[kind]["mean_vel"].append(m.mean_velocity_error)
            per_system[kind]["final_pos"].append(m.final_position_drift)
            per_system[kind]["mean_pos"].append(m.mean_position_drift)
            print(f"   {kind:<11} final_vel={m.final_velocity_error:.4f}  "
                  f"final_pos={m.final_position_drift:.4f}")

    summary: dict = {"n_windows": N_WINDOWS, "outage_s": OUTAGE_S,
                      "seed": SEED, "start_idxs": starts,
                      "start_fracs": fracs, "by_system": {}}
    for kind, data in per_system.items():
        arr = np.array(data["final_vel"])
        summary["by_system"][kind] = {
            "final_vel_mean": float(arr.mean()),
            "final_vel_std":  float(arr.std()),
            "final_vel_p50":  percentile(arr, 50),
            "final_vel_p95":  percentile(arr, 95),
            "final_vel_max":  float(arr.max()),
            "final_vel_min":  float(arr.min()),
            "raw_final_vel":  data["final_vel"],
            "raw_final_pos":  data["final_pos"],
            "raw_mean_vel":   data["mean_vel"],
            "raw_mean_pos":   data["mean_pos"],
        }

    print(f"\n=== Summary (final velocity error, m/s, N={N_WINDOWS}) ===")
    print(f"{'system':<12}  {'mean':>8}  {'std':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}")
    for kind in ("vanilla", "rls", "continuous"):
        s = summary["by_system"][kind]
        print(f"{kind:<12}  {s['final_vel_mean']:>8.4f}  {s['final_vel_std']:>8.4f}  "
              f"{s['final_vel_p50']:>8.4f}  {s['final_vel_p95']:>8.4f}  {s['final_vel_max']:>8.4f}")

    out_path = RESULTS_DIR / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
