"""
Standalone script: run Proposed DDPG WITHOUT warm-start for 600 episodes
to demonstrate DRL convergence from high → low delay.

Saves results to results_convergence/ and overlays with existing
LatencyMin / EnergyMin / NoCache baselines from results/.

Usage:
    python3 -m examples.vec_caching.run_convergence
"""

import os, json, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Build infrastructure identical to main.py ──────────────────────────────
from examples.vec_caching.settings import (
    NUM_EDGES, NUM_VEHICLES, REPLAY_SIZE,
    USE_SUMO, T_END, NOISE_STD_INIT, NOISE_DECAY
)
from examples.vec_caching.infrastructure import (
    EdgeServer, Vehicle, CloudServer,
    LinkV2I, LinkEdgeEdge, LinkEdgeCloud
)
from examples.vec_caching.mobility import MobilityManager
from examples.vec_caching.ddpg_agent import DDPGAgent, STATE_DIM, ACTION_DIM
from examples.vec_caching.orchestrator import VECOrchestrator
from leaf.infrastructure import Infrastructure

if USE_SUMO:
    from examples.vec_caching.mobility import SumoMobilityManager as MobilityManager


NUM_EPISODES = 600          # more episodes to show convergence
RESULTS_SRC  = "examples/vec_caching/results"
RESULTS_CONV = "examples/vec_caching/results_convergence"
WINDOW       = 15           # smoothing window for plot


def _save(path, name, delays, energies):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"metrics_{name}.json"), "w") as f:
        json.dump({"delays": delays, "energies": energies}, f)


def _init_vehicle_cache(v):
    v.cached_services = []


def _init_edge_cache(e):
    e.cached_services = []


from examples.vec_caching.main import build_infrastructure as build_infra


def run_no_warmstart():
    print(f"\n{'='*60}")
    print(f"  Proposed DDPG  (NO warm-start, {NUM_EPISODES} episodes)")
    print(f"{'='*60}")

    infra, vehicles, edge_servers, cloud = build_infra()
    mob = MobilityManager(vehicles, edge_servers)

    for v in vehicles:
        _init_vehicle_cache(v)
    for e in edge_servers:
        _init_edge_cache(e)  # start completely empty

    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)

    # NO warm_start_cache — agent starts from random empty caches
    orch = VECOrchestrator(
        infra, vehicles, edge_servers, cloud, mob, agent,
        training=True, warm_start_cache=False
    )

    delays, energies = [], []
    for ep in range(NUM_EPISODES):
        d, e = orch.run_episode()
        delays.append(sum(d))
        energies.append(sum(e))
        agent.decay_noise()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  Ep {ep+1:4d}/{NUM_EPISODES} | "
                  f"Delay: {delays[-1]:.3f} s | Noise: {agent.noise_std:.3f}")

    _save(RESULTS_CONV, "Proposed", delays, energies)
    return delays, energies


def smooth(x, w=WINDOW):
    import pandas as pd
    return list(pd.Series(x).rolling(w, min_periods=1).mean())

def plot_convergence():
    # Load no-warmstart Proposed
    with open(os.path.join(RESULTS_CONV, "metrics_Proposed.json")) as f:
        prop = json.load(f)

    # Load existing baselines (LatencyMin, EnergyMin, NoCache)
    baselines = {}
    for name in ["LatencyMin", "EnergyMin", "NoCache"]:
        fpath = os.path.join(RESULTS_SRC, f"metrics_{name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                baselines[name] = json.load(f)

    styles = {
        "Proposed":   ("DDPG-based edge caching and offloading", "red",   "-"),
        "LatencyMin": ("Offloading based on latency minimization","blue",  "-"),
        "EnergyMin":  ("Offloading based on energy minimization", "black", "-"),
        "NoCache":    ("Offloading without edge caching",         "green", "-"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Fig. 5: Total delay per episode (no warm-start)", fontsize=13)

    episodes_prop = range(1, len(prop["delays"]) + 1)
    label, color, ls = styles["Proposed"]
    ax.plot(episodes_prop, smooth(prop["delays"]),
            color=color, linestyle=ls, linewidth=1.5, label=label)

    for name, data in baselines.items():
        label, color, ls = styles[name]
        ep = range(1, len(data["delays"]) + 1)
        ax.plot(ep, smooth(data["delays"]),
                color=color, linestyle=ls, linewidth=1.5, label=label)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total delay per episode", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out = os.path.join(RESULTS_CONV, "figure_5_convergence.png")
    fig.savefig(out, dpi=150)
    print(f"\n  Saved → {out}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        plot_convergence()
    else:
        run_no_warmstart()
        plot_convergence()
