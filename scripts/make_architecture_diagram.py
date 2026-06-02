"""Render the system architecture diagram for the README.

Pure matplotlib (no external graphics tooling). Vertical-stack layout to
avoid horizontal label-arrow overlap.

Output:
  docs/figures/architecture.png  (high-dpi raster)
  docs/figures/architecture.svg  (vector)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

C_INPUT  = "#e8eef5"
C_LSTM   = "#1f77b4"
C_HEAD   = "#ff7f0e"
C_FILT   = "#9c27b0"
C_OUT    = "#2ca02c"
C_GPS    = "#ffb86b"


def box(ax, x, y, w, h, title, sub=None, fill="#ffffff", ec="#222", alpha=1.0,
        title_color="#111111", sub_color="#444", lw=1.5,
        title_size=12, sub_size=9.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=lw, facecolor=fill, edgecolor=ec, alpha=alpha)
    ax.add_patch(rect)
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title,
                ha="center", va="center", fontsize=title_size, weight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.28, sub,
                ha="center", va="center", fontsize=sub_size, color=sub_color,
                style="italic", linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center", fontsize=title_size, weight="bold", color=title_color)


def verify_text_fits(fig, ax, box_pad: float = 0.08, tol: float = 0.02) -> list[dict]:
    """Check that every Text artist whose center falls inside a FancyBboxPatch
    is fully contained within that box (accounting for FancyBboxPatch pad).

    Returns a list of overflow records. Empty list = clean layout.

    Uses matplotlib's renderer to get true text extents in data coords —
    catches font-metric-dependent overflows that eyeballing the source misses.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    boxes = []
    for patch in ax.patches:
        if isinstance(patch, mpatches.FancyBboxPatch):
            x, y = patch.get_x(), patch.get_y()
            w, h = patch.get_width(), patch.get_height()
            # Visual edges of FancyBboxPatch extend `pad` beyond the rect
            boxes.append((x - box_pad, y - box_pad,
                          x + w + box_pad, y + h + box_pad, patch))

    overflows = []
    for txt in ax.texts:
        if not txt.get_text():
            continue
        disp_bbox = txt.get_window_extent(renderer=renderer)
        x0, y0 = inv.transform((disp_bbox.x0, disp_bbox.y0))
        x1, y1 = inv.transform((disp_bbox.x1, disp_bbox.y1))
        tx0, ty0, tx1, ty1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2

        # Find the smallest box whose visual rect contains this text's center
        containing = None
        for bx0, by0, bx1, by1, patch in boxes:
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                area = (bx1 - bx0) * (by1 - by0)
                if containing is None or area < containing[0]:
                    containing = (area, bx0, by0, bx1, by1, patch)
        if containing is None:
            continue

        _, bx0, by0, bx1, by1, patch = containing
        overflow_left   = max(0.0, bx0 - tx0)
        overflow_right  = max(0.0, tx1 - bx1)
        overflow_top    = max(0.0, ty1 - by1)
        overflow_bottom = max(0.0, by0 - ty0)
        if max(overflow_left, overflow_right, overflow_top, overflow_bottom) > tol:
            overflows.append({
                "text": txt.get_text().replace("\n", "\\n"),
                "text_bbox": (round(tx0, 3), round(ty0, 3), round(tx1, 3), round(ty1, 3)),
                "box_bbox":  (round(bx0, 3), round(by0, 3), round(bx1, 3), round(by1, 3)),
                "overflow":  {
                    "left":   round(overflow_left,   3),
                    "right":  round(overflow_right,  3),
                    "top":    round(overflow_top,    3),
                    "bottom": round(overflow_bottom, 3),
                },
            })
    return overflows


def arrow(ax, x1, y1, x2, y2, color="#333", lw=2.0, ls="-",
          label=None, label_color="#333", label_size=10, label_offset=(0.25, 0)):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.7",
                        lw=lw, color=color, linestyle=ls),
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=label_size, color=label_color, style="italic")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(16, 11), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(8.5, 12.5, "GPS-Denied Navigation Pipeline",
            ha="center", fontsize=17, weight="bold", color="#111111")
    ax.text(8.5, 12.05,
            "Frozen LSTM body  ·  RLS-adapted linear head  ·  Velocity-only Kalman filter  ·  3-D position estimate",
            ha="center", fontsize=11, color="#444444", style="italic")

    # Vertical pipeline x positions — wider main column so long subtitles fit on one line
    cx = 8.5     # center column
    bw = 8.0     # default box width
    bx = cx - bw / 2  # = 4.5

    # ROW 1: IMU input
    box(ax, bx, 10.3, bw, 1.1,
        "IMU @ 200 Hz",
        sub="6 channels  ·  gyro x,y,z  +  accel x,y,z",
        fill=C_INPUT, ec="#111111")

    # ROW 2: LSTM body
    arrow(ax, cx, 10.3, cx, 9.5, color="#666",
          label="raw IMU stream", label_offset=(0.30, 0), label_size=9.5)
    box(ax, bx, 8.4, bw, 1.1,
        "LSTM body  (frozen)",
        sub="2 layers  ·  128 hidden  ·  202k params  ·  trained end-to-end on EuRoC",
        fill=C_LSTM, alpha=0.20, ec=C_LSTM, lw=2.0)

    # ROW 3: RLS head
    arrow(ax, cx, 8.4, cx, 7.6, color=C_LSTM,
          label="hidden features (128-D)", label_offset=(0.30, 0), label_size=9.5)
    box(ax, bx, 6.5, bw, 1.1,
        "RLS-adapted linear head",
        sub="128 → 3  ·  online updates while GPS is available",
        fill=C_HEAD, alpha=0.20, ec=C_HEAD, lw=2.0)

    # Side bus: GPS into RLS update.
    # All supervision info lives inside the GPS box (taller, 2-line subtitle),
    # so the dashed arrow can stay unlabeled and clean.
    gps_x, gps_w, gps_h = 0.4, 3.2, 1.4
    gps_y = 6.35   # centers GPS at y=7.05, same as RLS head, so arrow is horizontal
    gps_right = gps_x + gps_w  # = 3.6
    box(ax, gps_x, gps_y, gps_w, gps_h,
        "GPS (when available)",
        sub="ground-truth velocity\nsupervises RLS (pre-outage)",
        fill=C_GPS, alpha=0.35, ec="#cc7f0e", lw=1.5,
        title_size=11.5, sub_size=9.0)
    arrow(ax, gps_right, 7.05, bx, 7.05, color="#cc7f0e", lw=1.8, ls="--")

    # ROW 4: Filter
    arrow(ax, cx, 6.5, cx, 5.7, color=C_HEAD,
          label="velocity prediction (m/s)", label_offset=(0.30, 0), label_size=9.5)
    box(ax, bx, 4.6, bw, 1.1,
        "Velocity-only Kalman filter",
        sub="3-state  ·  σ_proc=0.5  ·  R from training residuals  ·  update @ 8 Hz",
        fill=C_FILT, alpha=0.18, ec=C_FILT, lw=2.0)

    # ROW 5: Output
    arrow(ax, cx, 4.6, cx, 3.8, color=C_FILT,
          label="smoothed velocity (m/s)", label_offset=(0.30, 0), label_size=9.5)
    box(ax, bx, 2.7, bw, 1.1,
        "3-D Position estimate",
        sub="trapezoidal integration  ·  drift metric: ‖p_pred − p_truth‖ at outage end",
        fill=C_OUT, alpha=0.22, ec=C_OUT, lw=2.0)

    # --- Bottom strip: API one-liner ---
    api_x, api_w = 0.4, 15.2
    api_cx = api_x + api_w / 2  # = 8.0
    rect = mpatches.FancyBboxPatch(
        (api_x, 0.4), api_w, 1.8,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=1.0, facecolor="#f7f7f7", edgecolor="#bbb")
    ax.add_patch(rect)
    ax.text(api_x + 0.3, 1.95, "API — reusable in other projects",
            ha="left", va="center", fontsize=10, weight="bold", color="#444")
    ax.text(api_cx, 1.50,
            "from gps_denied_nav import NavPipeline, EuRoCSequence",
            ha="center", fontsize=10.0, color="#222",
            family="monospace")
    ax.text(api_cx, 1.15,
            "pipeline = NavPipeline(model=lstm_v15, adapter=RLSHead(...), filter=VelocityOnlyFilter())",
            ha="center", fontsize=10.0, color="#222",
            family="monospace")
    ax.text(api_cx, 0.80,
            "result = pipeline.run_outage(seq, outage_start, outage_end)   # → 0.259 m/s after 30 s on MH_05",
            ha="center", fontsize=10.0, color="#666",
            family="monospace", style="italic")

    # Verify text fits inside boxes BEFORE saving — catches font-metric
    # overflows that visual inspection might miss.
    overflows = verify_text_fits(fig, ax)
    if overflows:
        print(f"\n[layout] {len(overflows)} text overflow(s) detected:")
        for o in overflows:
            sides = ", ".join(f"{k}={v}" for k, v in o["overflow"].items() if v > 0)
            print(f"  - text={o['text']!r}")
            print(f"    text_bbox={o['text_bbox']}  box_bbox={o['box_bbox']}")
            print(f"    overflow: {sides}")
    else:
        print("[layout] OK — all text fits inside its containing box.")

    out_png = FIG_DIR / "architecture.png"
    out_svg = FIG_DIR / "architecture.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {out_svg.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
