"""Navigation eval for LSTM v15 + Test-Time Training (TTT) adaptation.

Pipeline:
  1. Load v15 LSTM + norm stats.
  2. For each (sequence, outage_duration):
     a. Snapshot model weights.
     b. Run K gradient steps over pre-outage IMU windows (TTT).
     c. Build a NavPipeline with the now-adapted model + optional RLS head.
     d. Run the outage, record metrics.
     e. Restore model weights.

Sweep CLI:
    --K 3,10,30        number of TTT steps
    --lr 5e-5,1e-5     inner-loop learning rates
    --freeze-layers 0  number of LSTM layers to freeze

Outputs:
    results/lstm_v15_ttt/<seq>_test_metrics.json
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
from gps_denied_nav.adaptation import RLSHead, TTTAdapter
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
    parser.add_argument("--K", default="10",
                        help="comma-separated TTT step counts to sweep")
    parser.add_argument("--lr", default="5e-5",
                        help="comma-separated TTT learning rates to sweep")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="number of LSTM layers to freeze")
    parser.add_argument("--with-rls", action="store_true",
                        help="combine TTT (adapts LSTM body) with RLS (adapts head)")
    parser.add_argument("--checkpoint", default="checkpoints/lstm_v15.pt")
    parser.add_argument("--start-frac", type=float, default=0.4)
    args = parser.parse_args()

    Ks = parse_list(args.K, int)
    lrs = parse_list(args.lr, float)
    outages = parse_list(args.outages, float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq = EuRoCSequence.load(args.seq, SEQUENCES_DIR)
    print(f"Sequence: {seq.name}  ({seq.n_samples} samples, {seq.duration_s:.1f}s)\n")

    model, norm = load_lstm_checkpoint(args.checkpoint, device)

    results = []
    for outage_s in outages:
        outage_start, outage_end = seq.outage_window(args.start_frac, outage_s)
        for K, lr in product(Ks, lrs):
            ttt = TTTAdapter(model, K=K, lr=lr,
                             freeze_lstm_layers=args.freeze_layers)
            tag = f"outage={outage_s:.0f}s  K={K}  lr={lr:.0e}  with_rls={args.with_rls}"
            print(f"--- {tag} ---")
            with ttt.adapt(seq.imu, seq.gt_vel, outage_start, norm, device,
                           verbose=False) as ttt_info:
                adapter = (RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
                           if args.with_rls else None)
                pipeline = NavPipeline(model=model, adapter=adapter,
                                       filter=VelocityOnlyFilter(),
                                       norm=norm, device=device, update_stride=25)
                ev = OutageEvaluator(seq, outage_start_frac=args.start_frac)
                result, metrics = ev.evaluate(pipeline, outage_duration_s=outage_s)
            print(f"  final_vel = {metrics.final_velocity_error:.4f}  "
                  f"mean_vel = {metrics.mean_velocity_error:.4f}  "
                  f"final_pos = {metrics.final_position_drift:.4f}")
            results.append({
                "sequence": args.seq, "outage_s": outage_s,
                "K": K, "lr": lr, "with_rls": args.with_rls,
                "freeze_lstm_layers": args.freeze_layers,
                **metrics.as_dict(),
                "ttt_initial_loss": ttt_info["initial_loss"],
                "ttt_final_loss": ttt_info["final_loss"],
            })

    out_dir = PROJECT_ROOT / "results" / "lstm_v15_ttt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.seq}_test_metrics.json"
    with out_path.open("w") as f:
        json.dump({"sweep": results}, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
