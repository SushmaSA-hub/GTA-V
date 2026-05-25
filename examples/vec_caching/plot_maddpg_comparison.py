"""
Publication-quality comparison plot: MADDPG vs Single-Agent DDPG vs all baselines.

Generates two figures:
  1. figure_7_maddpg_delay.png   — Total delay per episode (all schemes)
  2. figure_8_maddpg_energy.png  — Total energy per episode (all schemes)
  3. figure_maddpg_full.png      — Combined (2-panel) publication figure

Usage (from leaf/ root):
    python3 -m examples.vec_caching.plot_maddpg_comparison
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR       = "examples/vec_caching/results"
CONV_DIR          = "examples/vec_caching/results_convergence"
OUT_DIR           = RESULTS_DIR
WINDOW            = 15   # smoothing window


def smooth(x, w=WINDOW):
    return pd.Series(x).rolling(w, min_periods=1).mean().values


# ─── Load data ───────────────────────────────────────────────────────────────
SCHEMES = [
    # (key,         label,                                  color,     ls,    lw)
    ("MADDPG",    "MA-DDPG (proposed – multi-agent)",     "#E85D04", "-",   2.4),
    ("Proposed",  "Single-Agent DDPG (joint cache+offload)","#F48C06","-",  2.0),
    ("LatencyMin","Offloading: latency minimization",     "#2D6A4F", "--",  1.8),
    ("EnergyMin", "Offloading: energy minimization",      "#1B4332", "--",  1.8),
    ("LRU",       "LRU edge caching",                     "#8338EC", ":",   1.8),
    ("NoCache",   "No edge caching (baseline)",           "#023E8A", "-.",  1.8),
]

data = {}
for key, label, color, ls, lw in SCHEMES:
    # MADDPG and baselines are in results/; Proposed (convergence run) in results_convergence/
    folders = [RESULTS_DIR, CONV_DIR]
    for folder in folders:
        path = os.path.join(folder, f"metrics_{key}.json")
        if os.path.exists(path) and os.path.getsize(path) > 10:
            with open(path) as f:
                d = json.load(f)
            data[key] = (label, color, ls, lw, d)
            break

print(f"Loaded schemes: {list(data.keys())}")
n_ep = max(len(v[4]["delays"]) for v in data.values())

# ─── Summary statistics ───────────────────────────────────────────────────────
print("\n── Final-100-episode averages ──────────────────────────────────────")
for key, (label, color, ls, lw, d) in data.items():
    avg_d = np.mean(d["delays"][-100:])
    avg_e = np.mean(d["energies"][-100:])
    print(f"  {key:<12s} delay={avg_d:.3f}s  energy={avg_e:.4f}J")
print()

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.subplots_adjust(wspace=0.28)

for ax, metric, ylabel, title in [
    (axes[0], "delays",   "Total delay per episode (s)",  "Comparison of Total Delay"),
    (axes[1], "energies", "Total energy per episode (J)", "Comparison of Total Energy"),
]:
    for key, (label, color, ls, lw, d) in data.items():
        vals  = d[metric]
        xs    = np.arange(1, len(vals) + 1)
        svals = smooth(vals)
        ax.plot(xs, svals, color=color, linestyle=ls, linewidth=lw,
                label=label, alpha=0.92)

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, n_ep)
    ax.tick_params(labelsize=10)

# Annotate MADDPG advantage
if "MADDPG" in data and "Proposed" in data:
    ma_d  = np.mean(data["MADDPG"][4]["delays"][-100:])
    prop_d = np.mean(data["Proposed"][4]["delays"][-100:])
    axes[0].annotate(
        f"MADDPG ↑ {prop_d - ma_d:.2f}s vs SA-DDPG",
        xy=(n_ep * 0.65, ma_d - 0.3),
        fontsize=8, color="#E85D04",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E85D04", alpha=0.8)
    )

fig.suptitle(
    "MA-DDPG (Multi-Agent, CTDE) vs Single-Agent DDPG and Baselines\n"
    "VEC Joint Service Caching and Computation Offloading",
    fontsize=13, fontweight="bold", y=1.01
)

out_path = os.path.join(OUT_DIR, "figure_maddpg_full.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")

# ─── Individual figures ───────────────────────────────────────────────────────
for ax_idx, (metric, ylabel, fname) in enumerate([
    ("delays",   "Total delay per episode (s)",  "figure_7_maddpg_delay.png"),
    ("energies", "Total energy per episode (J)", "figure_8_maddpg_energy.png"),
]):
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for key, (label, color, ls, lw, d) in data.items():
        vals  = d[metric]
        xs    = np.arange(1, len(vals) + 1)
        ax2.plot(xs, smooth(vals), color=color, linestyle=ls, linewidth=lw,
                 label=label, alpha=0.92)
    ax2.set_xlabel("Training Episode", fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xlim(1, n_ep)
    p = os.path.join(OUT_DIR, fname)
    fig2.tight_layout()
    fig2.savefig(p, dpi=180, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig2)

plt.close(fig)
print("\nAll plots generated successfully.")
