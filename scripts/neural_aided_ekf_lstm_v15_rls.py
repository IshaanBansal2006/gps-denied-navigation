"""Navigation eval for LSTM v15 + velocity-only filter + RLS adaptation head.

This is the headline result (decision 029). The RLS head ingests pre-outage
GPS-aided velocity at every STRIDE-th sample to specialize the final linear
projection, then predicts during the outage with the adapted weights.

Refactored 2026-05-24 to use the public `gps_denied_nav` API — this script
went from 244 hand-rolled lines to ~60 by composing `NavPipeline(model,
adapter, filter)`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gps_denied_nav import EuRoCSequence, NavPipeline, OutageEvaluator
from gps_denied_nav.adaptation import RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter
from gps_denied_nav.models import load_lstm_checkpoint

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60",
                        help="comma-separated outage durations (s)")
    parser.add_argument("--forgetting", type=float, default=0.995,
                        help="RLS forgetting factor (default 0.995)")
    parser.add_argument("--p-init", type=float, default=0.1,
                        help="RLS initial covariance diagonal — controls how aggressively "
                             "the head adapts. 0.1 sweet-spot for MH_05 30s (decision 029).")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_v15.pt",
                        help="path to the trained LSTM checkpoint")
    parser.add_argument("--start-frac", type=float, default=0.4,
                        help="outage start as fraction of sequence duration")
    args = parser.parse_args()

    outages = [float(x) for x in args.outages.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(args.seq, SEQUENCES_DIR)
    model, norm = load_lstm_checkpoint(args.checkpoint, device)
    print(f"Loaded {Path(args.checkpoint).name}: "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"Sequence: {seq.name}  ({seq.n_samples} samples, {seq.duration_s:.1f}s)")
    print(f"RLS: forgetting={args.forgetting}  p_init={args.p_init}\n")

    ev = OutageEvaluator(seq, outage_start_frac=args.start_frac)
    results = []
    for outage_s in outages:
        adapter = RLSHead(in_dim=model.hidden_size, out_dim=3,
                           forgetting=args.forgetting, p_init=args.p_init)
        pipeline = NavPipeline(model=model, adapter=adapter,
                                filter=VelocityOnlyFilter(),
                                norm=norm, device=device, update_stride=25)
        _, metrics = ev.evaluate(pipeline, outage_duration_s=outage_s)
        print(f"  outage={outage_s:5.1f}s  "
              f"final_vel={metrics.final_velocity_error:.4f}  "
              f"mean_vel={metrics.mean_velocity_error:.4f}  "
              f"final_pos={metrics.final_position_drift:.4f}  "
              f"  ({adapter.n_updates} RLS updates)")
        results.append({"outage_s": outage_s,
                         **metrics.as_dict(),
                         "rls_n_updates": adapter.n_updates})

    out_dir = PROJECT_ROOT / "results" / "lstm_v15_rls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump({
            "sequence": args.seq, "checkpoint": args.checkpoint,
            "forgetting": args.forgetting, "p_init": args.p_init,
            "start_frac": args.start_frac,
            "by_outage": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
