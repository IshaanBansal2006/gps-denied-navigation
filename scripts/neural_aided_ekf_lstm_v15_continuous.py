"""Navigation eval for LSTM v15 + RLS pre-outage + Continuous adaptation during outage.

Pipeline:
  1. Load v15 LSTM + norm stats.
  2. For each (sequence, outage_duration):
     a. Initialize RLSHead + ContinuousAdapter (wrapping the same RLS).
     b. Pre-outage warmup updates RLS with GPS-aided targets (normal RLS).
     c. During outage, continuous_adapter.update_during_outage() runs at every
        STRIDE step using (smoothed filter velocity, gyro-rotated prev velocity)
        as the pseudo-target.

Outputs:
    results/lstm_v15_continuous/<seq>_test_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gps_denied_nav import EuRoCSequence, NavPipeline, OutageEvaluator
from gps_denied_nav.adaptation import ContinuousAdapter, RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter
from gps_denied_nav.models import load_lstm_checkpoint

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"


def parse_list(s: str, cast=float):
    return [cast(x) for x in s.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="30",
                        help="comma-separated outage durations (s)")
    parser.add_argument("--alpha-smooth", default="0.7",
                        help="convex weight on smoothed-filter pseudo-target")
    parser.add_argument("--ema-alpha", default="0.95",
                        help="EMA weight on smoothed filter velocity")
    parser.add_argument("--outage-lambda", default="0.999",
                        help="RLS forgetting factor during outage")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_v15.pt")
    parser.add_argument("--start-frac", type=float, default=0.4)
    args = parser.parse_args()

    alphas = parse_list(args.alpha_smooth)
    emas = parse_list(args.ema_alpha)
    olams = parse_list(args.outage_lambda)
    outages = parse_list(args.outages)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(args.seq, SEQUENCES_DIR)
    print(f"Sequence: {seq.name}  ({seq.n_samples} samples, {seq.duration_s:.1f}s)\n")

    model, norm = load_lstm_checkpoint(args.checkpoint, device)

    results = []
    for outage_s in outages:
        for alpha, ema, olam in product(alphas, emas, olams):
            rls = RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
            cont = ContinuousAdapter(rls=rls, alpha_smooth=alpha,
                                      ema_alpha=ema, outage_lambda=olam)
            pipeline = NavPipeline(model=model, adapter=rls,
                                   filter=VelocityOnlyFilter(),
                                   norm=norm, device=device, update_stride=25,
                                   continuous_adapter=cont)
            ev = OutageEvaluator(seq, outage_start_frac=args.start_frac)
            _, metrics = ev.evaluate(pipeline, outage_duration_s=outage_s)
            tag = f"outage={outage_s:.0f}s  alpha={alpha:.2f}  ema={ema:.2f}  lam={olam:.3f}"
            print(f"--- {tag} ---")
            print(f"  final_vel={metrics.final_velocity_error:.4f}  "
                  f"mean_vel={metrics.mean_velocity_error:.4f}  "
                  f"final_pos={metrics.final_position_drift:.4f}")
            results.append({
                "sequence": args.seq, "outage_s": outage_s,
                "alpha_smooth": alpha, "ema_alpha": ema, "outage_lambda": olam,
                **metrics.as_dict(),
            })

    out_dir = PROJECT_ROOT / "results" / "lstm_v15_continuous"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.seq}_test_metrics.json"
    with out_path.open("w") as f:
        json.dump({"sweep": results}, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
