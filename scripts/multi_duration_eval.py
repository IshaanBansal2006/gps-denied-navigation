"""Multi-duration outage sweep — how does each system degrade as the
outage gets longer?

Runs three systems × seven outage durations on MH_05 from a single
checkpoint. Produces a unified results JSON consumed by
``make_duration_curve_figure.py``.

Systems:
  - vanilla v15 + filter
  - v15 + filter + RLS                (decision 029)
  - v15 + filter + continuous adapter (decision 032)

Durations (s):
  5, 10, 15, 20, 30, 45, 60

The 30 s point is the published headline (decision 029). The shorter
durations test whether the adapters' setup cost is worth it on quick
outages (urban canyon flashes), and the longer ones test asymptotic
degradation (extended jamming).

Output:
  results/multi_duration/<checkpoint_stem>_<seq>.json
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
DEFAULT_DURATIONS = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)


def make_pipeline(kind: str, model, norm, device):
    if kind == "vanilla":
        return NavPipeline(model=model, adapter=None,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25)
    rls = RLSHead(in_dim=model.hidden_size, out_dim=3,
                   forgetting=0.995, p_init=0.1)
    if kind == "rls":
        return NavPipeline(model=model, adapter=rls,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25)
    if kind == "continuous":
        cont = ContinuousAdapter(rls=rls, alpha_smooth=0.0,
                                  ema_alpha=0.95, outage_lambda=1.0)
        return NavPipeline(model=model, adapter=rls,
                            filter=VelocityOnlyFilter(),
                            norm=norm, device=device, update_stride=25,
                            continuous_adapter=cont)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="MH_05_difficult")
    parser.add_argument("--durations", default=",".join(str(d) for d in DEFAULT_DURATIONS),
                        help="comma-separated outage durations (s)")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_v15.pt")
    parser.add_argument("--start-frac", type=float, default=0.4)
    args = parser.parse_args()

    durations = [float(x) for x in args.durations.split(",")]
    ckpt_path = Path(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(args.seq, SEQUENCES_DIR)
    model, norm = load_lstm_checkpoint(ckpt_path, device)
    print(f"Loaded {ckpt_path.name}: hidden_size={model.hidden_size} "
          f"num_layers={model.num_layers} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Sequence: {seq.name}  durations: {durations}\n")

    ev = OutageEvaluator(seq, outage_start_frac=args.start_frac)
    out_by_system: dict = {"vanilla": {}, "rls": {}, "continuous": {}}

    for outage_s in durations:
        print(f"--- {outage_s:.0f}s outage ---")
        for kind in ("vanilla", "rls", "continuous"):
            pipeline = make_pipeline(kind, model, norm, device)
            _, metrics = ev.evaluate(pipeline, outage_duration_s=outage_s)
            out_by_system[kind][str(outage_s)] = metrics.as_dict()
            print(f"  {kind:<11} final_vel={metrics.final_velocity_error:.4f}  "
                  f"final_pos={metrics.final_position_drift:.4f}")

    out_dir = PROJECT_ROOT / "results" / "multi_duration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ckpt_path.stem}_{args.seq}.json"
    payload = {
        "sequence": args.seq,
        "checkpoint": str(ckpt_path),
        "start_frac": args.start_frac,
        "durations": durations,
        "by_system": out_by_system,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
