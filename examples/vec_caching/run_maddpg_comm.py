"""
Run MA-DDPG-Comm: MA-DDPG with Explicit Communication Protocol.

Adds per-agent MessageEncoder + CommNet-style intent aggregation + conflict penalty
on top of the base CTDE MA-DDPG framework.

Usage (from leaf/ root):
    OMP_NUM_THREADS=1 python3 -m examples.vec_caching.run_maddpg_comm
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
from examples.vec_caching.maddpg_comm_agent import (
    MADDPGCommSystem, LOCAL_STATE_DIM, LOCAL_ACTION_DIM, NV_PER_EDGE
)
from examples.vec_caching.settings import (
    NUM_EDGES, NUM_SERVICES, NUM_VEHICLES, T_END,
    SINR_DB_MIN, SINR_DB_MAX, BANDWIDTH, MAX_SERVICES_EDGE,
)

RESULTS_DIR  = "examples/vec_caching/results"
CONV_DIR     = "examples/vec_caching/results_convergence"
NUM_EPISODES = 600


# ─────────────────────────────────────────────────────────────────────────────
# State builder (identical to run_maddpg.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_local_state(edge, vehicles, sinr_linear):
    local_veh = list(edge.vehicles_in_range)
    while len(local_veh) < NV_PER_EDGE:
        local_veh.append(local_veh[0] if local_veh else vehicles[0])
    local_veh = local_veh[:NV_PER_EDGE]

    n_tasks  = max(len(edge.vehicles_in_range), 1)
    bw_norm  = min((BANDWIDTH / n_tasks) / BANDWIDTH, 1.0)
    sinr_norm = min((sinr_linear - 1.0) / 5.0, 1.0)

    state = []
    for v in local_veh:
        sk = v.current_task_service if v.current_task_service is not None else 0
        one_hot = [0.0] * NUM_SERVICES
        one_hot[sk] = 1.0
        state.extend(one_hot)
        state.extend([1.0 if k in v.cached_services else 0.0
                      for k in range(NUM_SERVICES)])
        state.append(bw_norm)
        state.append(sinr_norm)
    state.extend([1.0 if k in edge.cached_services else 0.0
                  for k in range(NUM_SERVICES)])
    return np.array(state, dtype=np.float32)


def apply_local_action(action, edge, vehicles):
    scores = action[:NUM_SERVICES]
    ranked = sorted(range(NUM_SERVICES), key=lambda k: scores[k], reverse=True)
    edge.cached_services = ranked[:MAX_SERVICES_EDGE]

    offloads = {}
    for i, v in enumerate(vehicles[:NV_PER_EDGE]):
        idx = NUM_SERVICES + i * 2
        offloads[v] = (
            float(np.clip(action[idx],     0.0, 1.0)),
            float(np.clip(action[idx + 1], 0.0, 1.0)),
        )
    return offloads


# ─────────────────────────────────────────────────────────────────────────────
# One episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(system, vehicles, edge_servers, cloud, mobility):
    sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
    sinr    = 10 ** (sinr_db / 10)
    delays, energies = [], []

    for _ in range(T_END):
        for v in vehicles:
            v.current_task_service = sample_task_service()

        # Build local states
        local_states = [build_local_state(e, vehicles, sinr)
                        for e in edge_servers]

        # ── Step 1-3: Communication + Action ──────────────────────────────────
        # Each agent encodes an intent message, broadcasts it, aggregates
        # neighbours' messages, then acts on the augmented observation.
        actions = system.get_actions(local_states)

        # ── Step 4: Apply actions ─────────────────────────────────────────────
        all_offloads = {}
        for i, (edge, action) in enumerate(zip(edge_servers, actions)):
            local_veh = list(edge.vehicles_in_range)
            while len(local_veh) < NV_PER_EDGE:
                local_veh.append(local_veh[0] if local_veh else vehicles[0])
            offloads = apply_local_action(action, edge, local_veh)
            all_offloads.update(offloads)

        # Compute delay & energy
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

        # ── Step 5: Reward with conflict-resolution penalty ───────────────────
        r_delay = -avg_delay
        r_coord = MADDPGCommSystem.conflict_penalty(actions, edge_servers)
        reward  = r_delay + r_coord

        # Next state
        mobility.step()
        sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
        sinr    = 10 ** (sinr_db / 10)
        next_states = [build_local_state(e, vehicles, sinr) for e in edge_servers]

        system.store(local_states, actions, reward, next_states, done=False)
        system.update()

    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  MA-DDPG-Comm: {NUM_EDGES} agents | MSG_DIM=16 | {NUM_EPISODES} episodes")
    print(f"  Protocol: MessageEncoder broadcast + conflict-resolution penalty")
    print(f"{'='*65}\n")

    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)

    system = MADDPGCommSystem(
        n_agents=NUM_EDGES,
        local_state_dim=LOCAL_STATE_DIM,
        local_action_dim=LOCAL_ACTION_DIM,
    )

    all_delays, all_energies = [], []
    for ep in range(NUM_EPISODES):
        d, e = run_episode(system, veh, edg, cld, mob)
        all_delays.append(sum(d))
        all_energies.append(sum(e))
        system.decay_noise()
        if ep % 50 == 0:
            print(f"  Ep {ep:4d}/{NUM_EPISODES} | Delay: {all_delays[-1]:.3f}s | "
                  f"Energy: {all_energies[-1]:.3f}J | Noise: {system.noise_std:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "metrics_MADDPG_Comm.json")
    with open(out_path, "w") as f:
        json.dump({"delays": all_delays, "energies": all_energies}, f)
    print(f"\nSaved: {out_path}")

    # ── Comparison plot ───────────────────────────────────────────────────────
    W = 15
    def smooth(x): return pd.Series(x).rolling(W, min_periods=1).mean().values

    SCHEMES = [
        ("MADDPG_Comm", "MA-DDPG-Comm (comm. protocol)", "#E85D04", "-",  2.4),
        ("MADDPG",      "MA-DDPG (CTDE only)",            "#F48C06", "-",  2.0),
        ("Proposed",    "Single-Agent DDPG",               "#2D6A4F", "--", 1.8),
        ("LatencyMin",  "Latency minimization",            "#1B4332", "--", 1.6),
        ("EnergyMin",   "Energy minimization",             "#023E8A", "--", 1.6),
        ("LRU",         "LRU edge caching",                "#8338EC", ":",  1.6),
        ("NoCache",     "No edge caching (baseline)",      "#6c757d", "-.", 1.6),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric, ylabel in [
        (axes[0], "delays",   "Total delay per episode (s)"),
        (axes[1], "energies", "Total energy per episode (J)"),
    ]:
        for key, label, color, ls, lw in SCHEMES:
            folder = CONV_DIR if key == "Proposed" else RESULTS_DIR
            path   = os.path.join(folder, f"metrics_{key}.json")
            if not os.path.exists(path) or os.path.getsize(path) < 10:
                continue
            with open(path) as f:
                data = json.load(f)
            vals = data[metric]
            ax.plot(range(1, len(vals)+1), smooth(vals),
                    color=color, linestyle=ls, linewidth=lw,
                    label=label, alpha=0.9)
        ax.set_xlabel("Training Episode", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(1, NUM_EPISODES)

    fig.suptitle(
        "MA-DDPG-Comm vs MA-DDPG vs Single-Agent DDPG vs Baselines\n"
        "Explicit Communication Protocol: MessageEncoder Broadcast + Conflict Penalty",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "figure_maddpg_comm_comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Comparison plot saved: {fig_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_comm = np.mean(all_delays[-100:])
    print(f"\n{'='*55}")
    for key, label, *_ in SCHEMES:
        folder = CONV_DIR if key == "Proposed" else RESULTS_DIR
        path   = os.path.join(folder, f"metrics_{key}.json")
        if not os.path.exists(path) or os.path.getsize(path) < 10:
            continue
        with open(path) as f:
            d = json.load(f)
        avg = np.mean(d["delays"][-100:])
        flag = " ◄ best" if key == "MADDPG_Comm" else ""
        print(f"  {label:<42s} {avg:.3f}s{flag}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
