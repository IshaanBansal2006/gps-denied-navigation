"""Render the batched-EKF CUDA kernel benchmark figure.

Reads the benchmark JSONs in results/batched_ekf/ (no recomputation):
  - forward throughput vs batch size for every backend (float32)
  - training step (fwd + bwd, 100-step rollout) throughput and peak GPU memory

Output:
  docs/figures/cuda_ekf_benchmark.png
  docs/figures/cuda_ekf_benchmark.svg
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "batched_ekf"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#ebeae6"

END_LABEL_OFFSET = {"torch_ref_cuda": 7, "cpp_cpu_threads": -7}

# Fixed categorical order; color follows the backend in every panel.
BACKENDS = [
    ("cuda_kernel", "Custom CUDA kernel", "#2a78d6", "o"),
    ("torch_ref_cuda", "PyTorch ops, GPU", "#eb6834", "s"),
    ("cpp_cpu_threads", "Same C++ code, CPU threads", "#1baf7a", "^"),
    ("torch_ref_cpu", "PyTorch ops, CPU", "#eda100", "D"),
    ("numpy_ekf15", "NumPy EKF15 loop", "#e87ba4", "v"),
]


def load_rows(name: str) -> List[dict]:
    return json.loads((RESULTS / name).read_text())["rows"]


def series(rows: List[dict], backend: str) -> Tuple[List[int], List[float]]:
    pts = sorted((r["n_filters"], r["filter_steps_per_s"]) for r in rows if r["backend"] == backend)
    return [p[0] for p in pts], [p[1] for p in pts]


def fmt_rate(v: float) -> str:
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def throughput_panel(ax: plt.Axes) -> None:
    refs = load_rows("bench_forward_float32.json")
    kernels = load_rows("bench_ab_forward_sparse_float32.json")
    sources: Dict[str, List[dict]] = {
        "cuda_kernel": kernels, "cpp_cpu_threads": kernels,
        "torch_ref_cuda": refs, "torch_ref_cpu": refs, "numpy_ekf15": refs,
    }
    for key, label, color, marker in BACKENDS:
        xs, ys = series(sources[key], key)
        if key == "numpy_ekf15":
            ax.axhline(ys[0], color=color, linewidth=2, linestyle=(0, (4, 3)))
            ax.text(70000, ys[0] * 1.35, f"{label} ({fmt_rate(ys[0])}, float64)", color=TEXT_2, fontsize=8.5, ha="right")
            continue
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(fmt_rate(ys[-1]), (xs[-1], ys[-1]), xytext=(8, END_LABEL_OFFSET.get(key, 0)), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.7, 2.6e5)
    ax.set_xlabel("filters in the batch (N)", color=TEXT_2, fontsize=10)
    ax.set_ylabel("filter-steps per second (float32)", color=TEXT_2, fontsize=10)
    ax.set_title("Forward step throughput", loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TEXT)


def bar_panel(ax: plt.Axes, values: List[float], title: str, ylabel: str, fmt) -> None:
    labels = ["autograd through\nPyTorch ops", "custom kernel\n(hand-written bwd)"]
    colors = [BACKENDS[1][2], BACKENDS[0][2]]
    bars = ax.bar([0, 1], values, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, values):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=TEXT)
    ax.set_xticks([0, 1], labels)
    ax.set_ylim(0, max(values) * 1.18)
    if fmt is fmt_rate:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_rate(v) if v else "0"))
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", colors=TEXT)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    ref = next(r for r in load_rows("bench_train_float32.json")
               if r["backend"] == "torch_ref_autograd" and r["n_filters"] == 8192)
    ker = next(r for r in load_rows("bench_ab_train_sparse_float32.json")
               if r["backend"] == "cuda_kernel_fn" and r["n_filters"] == 8192)

    fig = plt.figure(figsize=(13, 4.6), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.35)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax in axes:
        style_axis(ax)
    throughput_panel(axes[0])
    bar_panel(axes[1], [ref["filter_steps_per_s"], ker["filter_steps_per_s"]],
              "Training step speed", "filter-steps per second", fmt_rate)
    bar_panel(axes[2], [ref["peak_mem_mb"] / 1024, ker["peak_mem_mb"] / 1024],
              "Training peak GPU memory", "GB", lambda v: f"{v:.2f} GB")
    fig.text(0.985, -0.06, "Training panels: forward + backward through a 100-step rollout, N = 8,192 filters, float32",
             color=TEXT_2, fontsize=8.5, ha="right")
    fig.suptitle("Batched 15-state EKF — RTX 4070 Laptop GPU", x=0.07, ha="left", color=TEXT,
                 fontsize=13, fontweight="bold", y=1.02)

    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"cuda_ekf_benchmark.{ext}", dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {FIG_DIR / 'cuda_ekf_benchmark.png'}")


if __name__ == "__main__":
    main()
