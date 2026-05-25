"""
Run Multi-Agent DDPG (MADDPG) for 600 episodes and compare with single-agent Proposed DDPG.

Each of the NUM_EDGES edge servers has its own Actor (local observations).
A single shared centralised Critic sees the global state + all actions.

Usage:
    env OMP_NUM_THREADS=1 python3 -m examples.vec_caching.run_maddpg
"""

import os, json, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

from examples.vec_caching.main import build_infrastructure, make_mobility
from examples.vec_caching.infrastructure import compute_task_delay_and_energy
from examples.vec_caching.mobility import sample_task_service
from examples.vec_caching.maddpg_agent import (
    MADDPGSystem, LOCAL_STATE_DIM, LOCAL_ACTION_DIM, NV_PER_EDGE
)
from examples.vec_caching.settings import (
    NUM_EDGES, NUM_SERVICES, NUM_VEHICLES, T_END,
    SINR_DB_MIN, SINR_DB_MAX, BANDWIDTH, MAX_SERVICES_EDGE,
)

RESULTS_DIR = "examples/vec_caching/results"
NUM_EPISODES = 600


# ─────────────────────────────────────────────────────────────────────────────
# Local state builder (per edge server)
# ─────────────────────────────────────────────────────────────────────────────

def build_local_state(edge, edge_idx, vehicles, sinr_linear):
    """Build local state for one edge server agent."""
    # Get vehicles in range of this edge
    local_vehicles = list(edge.vehicles_in_range)
    # Pad/trim to NV_PER_EDGE
    while len(local_vehicles) < NV_PER_EDGE:
        local_vehicles.append(local_vehicles[0] if local_vehicles else vehicles[0])
    local_vehicles = local_vehicles[:NV_PER_EDGE]

    n_tasks = max(len(edge.vehicles_in_range), 1)
    bw_norm = min((BANDWIDTH / n_tasks) / BANDWIDTH, 1.0)
    sinr_norm = min((sinr_linear - 1.0) / 5.0, 1.0)

    state = []
    for v in local_vehicles:
        sk = v.current_task_service if v.current_task_service is not None else 0
        one_hot = [0.0] * NUM_SERVICES
        one_hot[sk] = 1.0
        state.extend(one_hot)
        state.extend([1.0 if k in v.cached_services else 0.0 for k in range(NUM_SERVICES)])
        state.append(bw_norm)
        state.append(sinr_norm)

    # Edge's own cache indicators
    state.extend([1.0 if k in edge.cached_services else 0.0 for k in range(NUM_SERVICES)])

    return np.array(state, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Action parser (per edge server)
# ─────────────────────────────────────────────────────────────────────────────

def apply_local_action(action, edge, local_vehicles):
    """Apply one agent's local action to its edge and vehicles."""
    # First NUM_SERVICES values → caching decision
    scores = action[:NUM_SERVICES]
    ranked = sorted(range(NUM_SERVICES), key=lambda k: scores[k], reverse=True)
    edge.cached_services = ranked[:MAX_SERVICES_EDGE]

    # Remaining 2*NV_PER_EDGE → offloading ratios
    offloads = {}
    for i, v in enumerate(local_vehicles[:NV_PER_EDGE]):
        idx = NUM_SERVICES + i * 2
        o_v = float(np.clip(action[idx],     0.0, 1.0))
        o_e = float(np.clip(action[idx + 1], 0.0, 1.0))
        offloads[v] = (o_v, o_e)
    return offloads


# ─────────────────────────────────────────────────────────────────────────────
# One MADDPG episode
# ─────────────────────────────────────────────────────────────────────────────

def run_maddpg_episode(system, vehicles, edge_servers, cloud, mobility):
    sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
    sinr = 10 ** (sinr_db / 10)

    delays, energies = [], []

    for _ in range(T_END):
        # Assign tasks
        for v in vehicles:
            v.current_task_service = sample_task_service()

        # Build local states for each agent
        local_states = [
            build_local_state(edge, i, vehicles, sinr)
            for i, edge in enumerate(edge_servers)
        ]

        # Get actions from MADDPG
        actions = system.get_actions(local_states)

        # Apply actions
        all_offloads = {}
        for i, (edge, action) in enumerate(zip(edge_servers, actions)):
            local_veh = list(edge.vehicles_in_range)
            while len(local_veh) < NV_PER_EDGE:
                local_veh.append(local_veh[0] if local_veh else vehicles[0])
            offloads = apply_local_action(action, edge, local_veh)
            all_offloads.update(offloads)

        # Compute delay and energy
        slot_delays, slot_energies = [], []
        for edge in edge_servers:
            n = len(edge.vehicles_in_range)
            if n == 0:
                continue
            for v in edge.vehicles_in_range:
                sk = v.current_task_service or 0
                o_v, o_e = all_offloads.get(v, (0.9, 0.0))
                T, E, _ = compute_task_delay_and_energy(
                    v, edge, cloud, sk, o_v, o_e,
                    n_tasks_at_edge=n, sinr_linear=sinr,
                    edge_servers=edge_servers,
                )
                slot_delays.append(T)
                slot_energies.append(E)

        avg_delay  = sum(slot_delays)  / max(len(slot_delays), 1)
        avg_energy = sum(slot_energies) / max(len(slot_energies), 1)
        delays.append(avg_delay)
        energies.append(avg_energy)

        # Reward = negative delay (cooperative, shared across agents)
        reward = -avg_delay

        # Build next local states
        mobility.step()
        sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
        sinr = 10 ** (sinr_db / 10)
        next_local_states = [
            build_local_state(edge, i, vehicles, sinr)
            for i, edge in enumerate(edge_servers)
        ]

        # Store transition and update
        system.store(local_states, actions, reward, next_local_states, done=False)
        system.update()

    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  MADDPG: {NUM_EDGES} agents | {NUM_EPISODES} episodes")
    print(f"{'='*60}\n")

    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)

    system = MADDPGSystem(
        n_agents=NUM_EDGES,
        local_state_dim=LOCAL_STATE_DIM,
        local_action_dim=LOCAL_ACTION_DIM,
    )

    all_delays, all_energies = [], []
    for ep in range(NUM_EPISODES):
        d, e = run_maddpg_episode(system, veh, edg, cld, mob)
        all_delays.append(sum(d))
        all_energies.append(sum(e))
        system.decay_noise()
        if ep % 50 == 0:
            print(f"  Ep {ep:4d}/{NUM_EPISODES} | Delay: {all_delays[-1]:.3f}s | "
                  f"Energy: {all_energies[-1]:.3f}J | Noise: {system.noise_std:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    maddpg_path = os.path.join(RESULTS_DIR, "metrics_MADDPG.json")
    with open(maddpg_path, "w") as f:
        json.dump({"delays": all_delays, "energies": all_energies}, f)
    print(f"\nSaved: {maddpg_path}")

    # ── Comparison plot ───────────────────────────────────────────────────────
    W = 15
    def smooth(x): return pd.Series(x).rolling(W, min_periods=1).mean().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    schemes = {}
    for name, label, color, ls in [
        ("Proposed",  "Single-Agent DDPG (Proposed)",      "red",    "-"),
        ("MADDPG",    "Multi-Agent DDPG (MADDPG)",         "orange", "-"),
        ("LatencyMin","Offloading: latency minimization",  "blue",   "--"),
        ("EnergyMin", "Offloading: energy minimization",   "black",  "--"),
        ("LRU",       "LRU edge caching",                  "purple", ":"),
        ("NoCache",   "No edge caching (baseline)",        "green",  "--"),
    ]:
        folder = "examples/vec_caching/results_convergence" if name == "Proposed" \
                 else RESULTS_DIR
        path = os.path.join(folder, f"metrics_{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        schemes[name] = (label, color, ls, data)

    for ax, metric, ylabel, title_sfx in [
        (axes[0], "delays",   "Total delay per episode (s)",  "Delay"),
        (axes[1], "energies", "Total energy per episode (J)", "Energy"),
    ]:
        for name, (label, color, ls, data) in schemes.items():
            vals = data[metric]
            ax.plot(range(1, len(vals)+1), smooth(vals),
                    color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"MADDPG vs Single-Agent DDPG — {title_sfx}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(0, NUM_EPISODES)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure_maddpg_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved: {out}")

    # Summary
    avg_s  = sum(all_delays[-100:]) / 100
    with open(os.path.join("examples/vec_caching/results_convergence",
                           "metrics_Proposed.json")) as f:
        prop = json.load(f)
    avg_p = sum(prop["delays"][-100:]) / 100
    print(f"\n{'='*50}")
    print(f"  Single-Agent DDPG avg (last 100 ep): {avg_p:.3f}s")
    print(f"  MADDPG           avg (last 100 ep): {avg_s:.3f}s")
    if avg_s < avg_p:
        print(f"  ✅ MADDPG WINS by {avg_p - avg_s:.3f}s improvement!")
    else:
        print(f"  ❌ Single-Agent DDPG is better by {avg_s - avg_p:.3f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
