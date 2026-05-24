"""Navigation eval for LSTM v18 + adapter combinations.

Runs the v18 checkpoint (bigger LSTM, curriculum-trained, val_final
selection — see decision 030) through three adapter configurations at
multiple outage durations on MH_05:

  1. Vanilla v18 + velocity-only filter         (no adapter)
  2. v18 + filter + RLS                          (decision 029 recipe)
  3. v18 + filter + continuous adaptation        (decision 032 recipe)

Pre-staged so it can fire the instant v18 training completes, without
having to write+test a fresh eval harness in the post-training cycle.

Output: results/lstm_v18_nav/test_metrics.json
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
from gps_denied_nav.adaptation import ContinuousAdapter, RLSHead
from gps_denied_nav.filters import VelocityOnlyFilter
from gps_denied_nav.models import load_lstm_checkpoint

SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--outages", default="5,10,30,60",
                        help="comma-separated outage durations (s)")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_v18.pt")
    parser.add_argument("--start-frac", type=float, default=0.4)
    args = parser.parse_args()

    outages = [float(x) for x in args.outages.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(args.seq, SEQUENCES_DIR)
    model, norm = load_lstm_checkpoint(args.checkpoint, device)
    print(f"Loaded {Path(args.checkpoint).name}: "
          f"hidden_size={model.hidden_size} num_layers={model.num_layers} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Sequence: {seq.name}  ({seq.n_samples} samples, {seq.duration_s:.1f}s)\n")

    ev = OutageEvaluator(seq, outage_start_frac=args.start_frac)
    by_outage = {}

    for outage_s in outages:
        print(f"--- {outage_s:.0f}s outage ---")
        row = {}

        # 1) Vanilla v18 + filter
        p_van = NavPipeline(model=model, adapter=None,
                             filter=VelocityOnlyFilter(),
                             norm=norm, device=device, update_stride=25)
        _, m_van = ev.evaluate(p_van, outage_duration_s=outage_s)
        row["vanilla"] = m_van.as_dict()
        print(f"  vanilla        final_vel={m_van.final_velocity_error:.4f}  "
              f"final_pos={m_van.final_position_drift:.4f}")

        # 2) v18 + filter + RLS
        rls = RLSHead(in_dim=model.hidden_size, out_dim=3,
                       forgetting=0.995, p_init=0.1)
        p_rls = NavPipeline(model=model, adapter=rls,
                             filter=VelocityOnlyFilter(),
                             norm=norm, device=device, update_stride=25)
        _, m_rls = ev.evaluate(p_rls, outage_duration_s=outage_s)
        row["rls"] = m_rls.as_dict()
        print(f"  rls            final_vel={m_rls.final_velocity_error:.4f}  "
              f"final_pos={m_rls.final_position_drift:.4f}")

        # 3) v18 + filter + continuous
        rls2 = RLSHead(in_dim=model.hidden_size, out_dim=3,
                        forgetting=0.995, p_init=0.1)
        cont = ContinuousAdapter(rls=rls2, alpha_smooth=0.0,
                                  ema_alpha=0.95, outage_lambda=1.0)
        p_cont = NavPipeline(model=model, adapter=rls2,
                              filter=VelocityOnlyFilter(),
                              norm=norm, device=device, update_stride=25,
                              continuous_adapter=cont)
        _, m_cont = ev.evaluate(p_cont, outage_duration_s=outage_s)
        row["continuous"] = m_cont.as_dict()
        print(f"  continuous     final_vel={m_cont.final_velocity_error:.4f}  "
              f"final_pos={m_cont.final_position_drift:.4f}")

        by_outage[str(outage_s)] = row

    out_dir = PROJECT_ROOT / "results" / "lstm_v18_nav"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump({
            "sequence": args.seq,
            "checkpoint": args.checkpoint,
            "start_frac": args.start_frac,
            "by_outage": by_outage,
        }, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
