"""Render the 'error vs outage duration' figure.

Two side-by-side panels:
  - Left: final velocity error vs outage duration, three systems.
  - Right: final position drift vs outage duration, same three systems.

Drawn from the JSON produced by ``multi_duration_eval.py``. Optionally
overlays a v18 curve if that JSON exists too — for direct v15-vs-v18
comparison across durations.

Output:
  docs/figures/duration_curve.png
  docs/figures/duration_curve.svg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "multi_duration"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

SYSTEM_STYLES = {
    "vanilla":    {"color": "#7a7a7a", "ls": "-",  "marker": "o", "label": "vanilla v15 + filter"},
    "rls":        {"color": "#1f77b4", "ls": "-",  "marker": "s", "label": "v15 + filter + RLS (headline)"},
    "continuous": {"color": "#2ca02c", "ls": "-",  "marker": "^", "label": "v15 + filter + continuous"},
}

V18_STYLES = {
    "vanilla":    {"color": "#7a7a7a", "ls": "--", "marker": "o", "label": "vanilla v18 + filter"},
    "rls":        {"color": "#1f77b4", "ls": "--", "marker": "s", "label": "v18 + filter + RLS"},
    "continuous": {"color": "#2ca02c", "ls": "--", "marker": "^", "label": "v18 + filter + continuous"},
}


def load_results(json_path: Path) -> dict:
    with json_path.open() as f:
        return json.load(f)


def plot_one_dataset(ax_vel, ax_pos, payload, styles):
    durations = payload["durations"]
    by_system = payload["by_system"]
    for kind, style in styles.items():
        if kind not in by_system:
            continue
        finals = [by_system[kind][str(d)]["final_velocity_error"] for d in durations]
        poss = [by_system[kind][str(d)]["final_position_drift"] for d in durations]
        ax_vel.plot(durations, finals, marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], lw=2.0, ms=7, label=style["label"])
        ax_pos.plot(durations, poss, marker=style["marker"], color=style["color"],
                    linestyle=style["ls"], lw=2.0, ms=7, label=style["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15-json", default="results/multi_duration/lstm_v15_MH_05_difficult.json")
    parser.add_argument("--v18-json", default="results/multi_duration/lstm_v18_MH_05_difficult.json",
                        help="path to v18 results — overlaid if file exists")
    parser.add_argument("--out", default="duration_curve")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    v15_path = PROJECT_ROOT / args.v15_json
    v18_path = PROJECT_ROOT / args.v18_json
    v15 = load_results(v15_path)
    print(f"Loaded {v15_path.relative_to(PROJECT_ROOT)}")
    v18 = load_results(v18_path) if v18_path.exists() else None
    if v18:
        print(f"Loaded {v18_path.relative_to(PROJECT_ROOT)} — overlaying")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
    })

    fig, (ax_vel, ax_pos) = plt.subplots(1, 2, figsize=(14, 6), dpi=110)

    plot_one_dataset(ax_vel, ax_pos, v15, SYSTEM_STYLES)
    if v18:
        plot_one_dataset(ax_vel, ax_pos, v18, V18_STYLES)

    ax_vel.set_xlabel("Outage duration (s)")
    ax_vel.set_ylabel("Final velocity error (m/s · lower is better)")
    ax_vel.set_title("Velocity error vs outage duration")
    ax_vel.legend(loc="best", frameon=False, fontsize=9.5)

    ax_pos.set_xlabel("Outage duration (s)")
    ax_pos.set_ylabel("Final position drift (m · lower is better)")
    ax_pos.set_title("Position drift vs outage duration")
    ax_pos.legend(loc="best", frameon=False, fontsize=9.5)

    fig.suptitle(
        f"Operating envelope on EuRoC {v15['sequence']}  ·  "
        "where each system wins as a function of outage duration",
        fontsize=13, y=1.02,
    )

    out_png = FIG_DIR / f"{args.out}.png"
    out_svg = FIG_DIR / f"{args.out}.svg"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {out_svg.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
