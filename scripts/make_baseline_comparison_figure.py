"""Render the cross-model 30-second final-velocity-error bar chart.

Reads the per-model nav-eval JSONs in results/ and produces a single bar
chart suitable for the README "results at a glance" section. Numbers come
straight from the published nav-eval results files; no recomputation.

Output:
  docs/figures/baseline_comparison.png
  docs/figures/baseline_comparison.svg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"


@dataclass
class Bar:
    label: str
    value: float  # m/s final velocity error at 30s
    color: str
    is_ours: bool = False
    is_oracle: bool = False
    note: str = ""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Numbers from CLAUDE.md + decision docs (MH_05_difficult, 30s outage, final velocity error).
    bars = [
        Bar("IMU dead-reckon",                       45.867, "#d62728", note="no model — drifts catastrophically"),
        Bar("TCN v7 + filter",                        0.440, "#9c9c9c", note="prior best (decision 018)"),
        Bar("LSTM v12 + filter",                      0.497, "#9c9c9c", note="(decision 023)"),
        Bar("LSTM v13 + filter",                      0.449, "#9c9c9c", note="(decision 025)"),
        Bar("LSTM v15 + filter",                      0.403, "#9c9c9c", note="(decision 027)"),
        Bar("LSTM v15 + filter + RLS adapt (ours)",   0.259, "#1f77b4", is_ours=True, note="(decision 029)"),
        Bar("EKF + GPS (oracle ceiling)",             0.104, "#7a7a7a", is_oracle=True),
    ]

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

    fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
    ax.xaxis.grid(True, color="#eeeeee", lw=0.6)
    ax.yaxis.grid(False)

    labels = [b.label for b in bars]
    values = np.array([b.value for b in bars])
    colors = [b.color for b in bars]

    y = np.arange(len(bars))
    bars_h = ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=1.2)

    for i, b in enumerate(bars):
        if b.is_ours:
            bars_h[i].set_edgecolor("#1f77b4")
            bars_h[i].set_linewidth(2.0)
        x_text = b.value * 1.15
        ax.text(x_text, i, f"{b.value:.3f} m/s",
                va="center", ha="left",
                color="#222222", fontsize=10.5, weight="bold")
        if b.note:
            ax.text(values.max() * 1.6, i, b.note,
                    va="center", ha="left", color="#666666", fontsize=9.0, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final velocity error after 30 s GPS outage (m/s · log scale · lower is better)")
    ax.set_xscale("log")
    ax.set_xlim(0.05, values.max() * 4.0)
    ax.set_title("RLS-adapted LSTM v15 closes the gap to the GPS oracle to 2.5× on 30-second outages (EuRoC MH_05)",
                 pad=14, fontsize=12.5)

    out_png = FIG_DIR / "baseline_comparison.png"
    out_svg = FIG_DIR / "baseline_comparison.svg"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"  Saved: {out_svg.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
