"""
Plot results from the VEC caching simulation — reproduces Figures 5–11 of the paper.

Usage (from leaf/ root):
    python -m examples.vec_caching.plot_results
    python -m examples.vec_caching.plot_results --results-dir examples/vec_caching/results
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCHEME_STYLES = {
    "Proposed":   {"color": "red",    "linestyle": "-",  "label": "DDPG-based edge caching and offloading"},
    "NoCache":    {"color": "green",  "linestyle": "-",  "label": "Offloading without edge caching"},
    "LatencyMin": {"color": "blue",   "linestyle": "-",  "label": "Offloading based on latency minimization"},
    "EnergyMin":  {"color": "black",  "linestyle": "-",  "label": "Offloading based on energy minimization"},
    "LRU":        {"color": "purple", "linestyle": "--", "label": "LRU edge caching and offloading"},
}


def load_metrics(results_dir: str) -> Dict[str, Dict]:
    metrics = {}
    for scheme in SCHEME_STYLES:
        path = os.path.join(results_dir, f"metrics_{scheme}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[scheme] = json.load(f)
    return metrics


def smooth(data: List[float], w: int = 5) -> np.ndarray:
    """Simple moving-average smoothing."""
    arr = np.array(data)
    kernel = np.ones(w) / w
    if len(arr) >= w:
        return np.convolve(arr, kernel, mode="valid")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Total Delay per Episode
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig5(metrics: Dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme, data in metrics.items():
        style = SCHEME_STYLES.get(scheme, {})
        delays = smooth(data["delays"], w=5)
        episodes = np.arange(1, len(delays) + 1)
        ax.plot(episodes, delays,
                color=style.get("color", "gray"),
                label=style.get("label", scheme),
                linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total delay per episode", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Fig. 5: Total delay per episode", fontsize=13)
    path = os.path.join(out_dir, "figure_5_delay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Total Energy per Episode
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig6(metrics: Dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme, data in metrics.items():
        style = SCHEME_STYLES.get(scheme, {})
        energies = smooth(data["energies"], w=5)
        episodes = np.arange(1, len(energies) + 1)
        ax.plot(episodes, energies,
                color=style.get("color", "gray"),
                label=style.get("label", scheme),
                linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total energy per episode (J)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Fig. 6: Total energy per episode", fontsize=13)
    path = os.path.join(out_dir, "figure_6_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Delay vs Vehicle Density (runs mini evaluation per density)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig7_from_file(out_dir: str):
    """Plot Fig 7 if density sweep results file exists."""
    path = os.path.join(out_dir, "density_results.json")
    if not os.path.exists(path):
        print(f"  Skipping Fig 7: {path} not found (run --density-sweep first)")
        return
    with open(path) as f:
        data = json.load(f)
    rhos = sorted(data.keys(), key=int)
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme in SCHEME_STYLES:
        if scheme not in data[rhos[0]]:
            continue
        style = SCHEME_STYLES[scheme]
        y = [data[r][scheme] for r in rhos]
        ax.plot([int(r) for r in rhos], y,
                marker="o", color=style["color"], label=style["label"])
    ax.set_xlabel("Vehicle density ρ", fontsize=12)
    ax.set_ylabel("Average delay (s)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Fig. 7: Delay vs Vehicle Density", fontsize=13)
    out = os.path.join(out_dir, "figure_7_density.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="examples/vec_caching/results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nLoading metrics from: {args.results_dir}")
    metrics = load_metrics(args.results_dir)
    if not metrics:
        print("  No metric files found. Run main.py first.")
    else:
        print(f"  Found metrics for: {list(metrics.keys())}")
        plot_fig5(metrics, args.results_dir)
        plot_fig6(metrics, args.results_dir)
        plot_fig7_from_file(args.results_dir)
        print("\n  All available figures generated.")
