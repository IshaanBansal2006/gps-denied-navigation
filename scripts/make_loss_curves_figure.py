"""Render the LSTM v15 training-dynamic figure for the README / writeup.

Plots train loss, val_mean rollout error, and val_final rollout error across
training epochs. Annotates the best-val_mean checkpoint (epoch 24, selected by
training script) and the eventual val_final minimum (~epoch 39) to illustrate
the checkpoint-selection-criterion observation from decision 027.

Output:
  docs/figures/loss_curves_v15.png
  docs/figures/loss_curves_v15.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS = PROJECT_ROOT / "results" / "lstm_v15" / "loss_history.json"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

C_TRAIN = "#7a7a7a"
C_VMEAN = "#1f77b4"
C_VFIN  = "#d62728"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with LOSS.open() as f:
        h = json.load(f)

    train = np.array(h["train_loss"])
    vmean = np.array(h["val_mean_err"])
    vfin  = np.array(h["val_final_err"])
    n = len(train)
    epochs = np.arange(1, n + 1)

    best_vmean_epoch = int(np.argmin(vmean) + 1)
    best_vfin_epoch  = int(np.argmin(vfin)  + 1)

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

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    ax.plot(epochs, train, color=C_TRAIN, lw=1.8, label="train loss (left axis)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (pure-nav, normalized vel space)", color=C_TRAIN)
    ax.tick_params(axis="y", labelcolor=C_TRAIN)

    ax2.plot(epochs, vmean, color=C_VMEAN, lw=2.2, label="val_mean (right axis)")
    ax2.plot(epochs, vfin,  color=C_VFIN,  lw=2.2, label="val_final (right axis)")
    ax2.set_ylabel("Validation rollout error (m/s)")

    ax.axvline(best_vmean_epoch, color=C_VMEAN, ls=":", lw=1.0, alpha=0.7)
    ax.axvline(best_vfin_epoch,  color=C_VFIN,  ls=":", lw=1.0, alpha=0.7)

    ax2.annotate(
        f"Selected checkpoint\n(min val_mean = {vmean[best_vmean_epoch-1]:.3f})\nepoch {best_vmean_epoch}",
        xy=(best_vmean_epoch, vmean[best_vmean_epoch - 1]),
        xytext=(best_vmean_epoch - 8, vmean[best_vmean_epoch - 1] + 0.18),
        fontsize=9.5, color=C_VMEAN,
        arrowprops=dict(arrowstyle="->", color=C_VMEAN, lw=0.9),
    )
    ax2.annotate(
        f"val_final minimum\n({vfin[best_vfin_epoch-1]:.3f})\nepoch {best_vfin_epoch}\n— 5–10% gain still on the table",
        xy=(best_vfin_epoch, vfin[best_vfin_epoch - 1]),
        xytext=(best_vfin_epoch - 17, vfin[best_vfin_epoch - 1] - 0.32),
        fontsize=9.5, color=C_VFIN,
        arrowprops=dict(arrowstyle="->", color=C_VFIN, lw=0.9),
    )

    lines = [l for l in ax.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", frameon=False, fontsize=10)

    ax.set_title("LSTM v15 training dynamic — checkpoint selection on val_mean leaves val_final headroom",
                 pad=12, fontsize=13)
    ax.set_xlim(1, n)

    out_png = FIG_DIR / "loss_curves_v15.png"
    out_svg = FIG_DIR / "loss_curves_v15.svg"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  train_loss span:  {train.min():.4f} – {train.max():.4f}")
    print(f"  val_mean span:    {vmean.min():.4f} – {vmean.max():.4f}  (min @ epoch {best_vmean_epoch})")
    print(f"  val_final span:   {vfin.min():.4f} – {vfin.max():.4f}  (min @ epoch {best_vfin_epoch})")
    print(f"  Saved: {out_png.relative_to(PROJECT_ROOT)}")
    print(f"  Saved: {out_svg.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
