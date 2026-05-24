"""Cross-sequence eval: does v15+RLS's 36% win replicate beyond MH_05?

Decision 029 selected the RLS adaptation hyperparameters (p_init=0.1, λ=0.995)
on the same test sequence (MH_05_difficult) it reported on — a flagged
limitation. This script tests whether the win generalizes by running the
same configurations on MH_03_medium, V1_03_difficult, MH_04_difficult.

For each sequence, runs three pipelines at the standard 30 s outage:
  - vanilla v15 + velocity-only filter (baseline)
  - v15 + filter + RLS (decision 029 system)
  - v15 + filter + continuous adapter (decision 032 system)

Output:
  results/cross_sequence/test_metrics.json
"""
from __future__ import annotations

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
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_sequence"

EVAL_SEQS = ["MH_03_medium", "V1_03_difficult", "MH_04_difficult", "MH_05_difficult"]
OUTAGE_S = 30.0
START_FRAC = 0.4


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, norm = load_lstm_checkpoint(PROJECT_ROOT / "checkpoints" / "lstm_v15.pt",
                                        device)
    all_results = {}

    for seq_name in EVAL_SEQS:
        try:
            seq = EuRoCSequence.load(seq_name, SEQUENCES_DIR)
        except FileNotFoundError:
            print(f"[skip] {seq_name}")
            continue

        ev = OutageEvaluator(seq, outage_start_frac=START_FRAC)
        seq_results = {}

        print(f"\n=== {seq_name} ===")

        # 1) Vanilla
        p_van = NavPipeline(model=model, adapter=None,
                             filter=VelocityOnlyFilter(),
                             norm=norm, device=device, update_stride=25)
        _, m_van = ev.evaluate(p_van, outage_duration_s=OUTAGE_S)
        seq_results["vanilla"] = m_van.as_dict()
        print(f"  vanilla        final={m_van.final_velocity_error:.4f}  "
              f"pos={m_van.final_position_drift:.4f}")

        # 2) RLS
        rls = RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
        p_rls = NavPipeline(model=model, adapter=rls,
                             filter=VelocityOnlyFilter(),
                             norm=norm, device=device, update_stride=25)
        _, m_rls = ev.evaluate(p_rls, outage_duration_s=OUTAGE_S)
        seq_results["rls"] = m_rls.as_dict()
        delta_rls = (m_rls.final_velocity_error - m_van.final_velocity_error
                     ) / m_van.final_velocity_error * 100
        print(f"  rls            final={m_rls.final_velocity_error:.4f}  "
              f"pos={m_rls.final_position_drift:.4f}  Δ={delta_rls:+.1f}%")

        # 3) Continuous
        rls2 = RLSHead(in_dim=128, out_dim=3, forgetting=0.995, p_init=0.1)
        cont = ContinuousAdapter(rls=rls2, alpha_smooth=0.0,
                                  ema_alpha=0.95, outage_lambda=1.0)
        p_cont = NavPipeline(model=model, adapter=rls2,
                              filter=VelocityOnlyFilter(),
                              norm=norm, device=device, update_stride=25,
                              continuous_adapter=cont)
        _, m_cont = ev.evaluate(p_cont, outage_duration_s=OUTAGE_S)
        seq_results["continuous"] = m_cont.as_dict()
        delta_cont = (m_cont.final_velocity_error - m_van.final_velocity_error
                      ) / m_van.final_velocity_error * 100
        print(f"  continuous     final={m_cont.final_velocity_error:.4f}  "
              f"pos={m_cont.final_position_drift:.4f}  Δ={delta_cont:+.1f}%")

        all_results[seq_name] = seq_results

    out_path = RESULTS_DIR / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump({
            "outage_s": OUTAGE_S,
            "start_frac": START_FRAC,
            "by_sequence": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
