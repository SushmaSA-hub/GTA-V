"""
Main simulation runner for the VEC caching paper implementation in LEAF.

All 6 comparison schemes from the paper:
  Proposed   — DDPG joint caching + offloading (minimize delay + energy)
  LatencyMin — DDPG offloading to minimize delay only
  EnergyMin  — DDPG offloading to minimize energy only
  Random     — Random caching and offloading each slot
  LRU        — LRU caching + latency-min offloading
  Cloud      — All tasks sent to cloud

Usage (from leaf/ root):
    python -m examples.vec_caching.main               # full run
    python -m examples.vec_caching.main --episodes 5  # quick smoke test
    python -m examples.vec_caching.main --scheme ddpg # proposed only
"""

import argparse
import json
import os
from typing import List

from leaf.infrastructure import Infrastructure

from examples.vec_caching.settings import (
    NUM_EDGES, NUM_VEHICLES, NUM_SERVICES, RHO,
    EDGE_COMM_RANGE, ROAD_LENGTH, NUM_EPISODES, REPLAY_SIZE,
    USE_SUMO
)
from examples.vec_caching.infrastructure import (
    Vehicle, EdgeServer, CloudServer,
    LinkV2I, LinkEdgeEdge, LinkEdgeCloud,
)
from examples.vec_caching.mobility import (
    MobilityManager, SumoMobilityManager,
    init_vehicle_cache, init_edge_cache
)
from examples.vec_caching.ddpg_agent import DDPGAgent, STATE_DIM, ACTION_DIM
from examples.vec_caching.orchestrator import VECOrchestrator
from examples.vec_caching.baselines import (
    run_nocache_episode,
    run_latencymin_ddpg_episode,
    run_energymin_ddpg_episode,
    run_random_episode,
    run_lru_episode,
    run_cloud_episode,
)


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_infrastructure(num_vehicles: int = NUM_VEHICLES, rho: int = RHO):
    """Build the LEAF infrastructure graph for the VEC scenario."""
    infra = Infrastructure()

    cloud = CloudServer()
    infra.add_node(cloud)

    edge_servers: List[EdgeServer] = []
    for i in range(NUM_EDGES):
        pos = (i + 0.5) * EDGE_COMM_RANGE
        edge = EdgeServer(position=pos)
        edge_servers.append(edge)
        infra.add_node(edge)

    for edge in edge_servers:
        infra.add_link(LinkEdgeCloud(edge, cloud))

    for i, e1 in enumerate(edge_servers):
        for j, e2 in enumerate(edge_servers):
            if i != j:
                infra.add_link(LinkEdgeEdge(e1, e2))

    vehicles: List[Vehicle] = []
    for _ in range(num_vehicles):
        v = Vehicle(position=0.0)
        vehicles.append(v)
        infra.add_node(v)

    for v in vehicles:
        for e in edge_servers:
            infra.add_link(LinkV2I(v, e))

    return infra, vehicles, edge_servers, cloud


def make_mobility(vehicles, edge_servers):
    if USE_SUMO:
        return SumoMobilityManager(vehicles, edge_servers)
    else:
        return MobilityManager(vehicles, edge_servers)


# ─────────────────────────────────────────────────────────────────────────────
# Scheme 1 (Proposed): DDPG joint caching + offloading (reward = -delay)
# ─────────────────────────────────────────────────────────────────────────────

def run_proposed(num_episodes: int, results_dir: str):
    print(f"\n{'='*60}")
    print(f"  Proposed: DDPG Joint Caching + Offloading")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}")

    infra, vehicles, edge_servers, cloud = build_infrastructure()
    mobility = make_mobility(vehicles, edge_servers)

    for v in vehicles:
        init_vehicle_cache(v)
    for e in edge_servers:
        init_edge_cache(e)

    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)
    orch = VECOrchestrator(
        infra, vehicles, edge_servers, cloud, mobility, agent,
        training=True, warm_start_cache=False  # NO warm-start to show convergence curve
    )

    all_delays, all_energies = [], []
    for ep in range(num_episodes):
        delays, energies = orch.run_episode()
        all_delays.append(sum(delays))
        all_energies.append(sum(energies))
        agent.decay_noise()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Ep {ep+1:4d}/{num_episodes} | "
                  f"Delay: {all_delays[-1]:.3f} s | "
                  f"Energy: {all_energies[-1]:.4e} J | "
                  f"Noise: {agent.noise_std:.3f}")

    os.makedirs(results_dir, exist_ok=True)
    agent.save(os.path.join(results_dir, "ddpg_proposed.pth"))
    _save(results_dir, "Proposed", all_delays, all_energies)
    return all_delays, all_energies


# ─────────────────────────────────────────────────────────────────────────────
# Scheme 2: Latency Minimization via DDPG (reward = -delay only)
# ─────────────────────────────────────────────────────────────────────────────

def run_latencymin(num_episodes: int, results_dir: str):
    print(f"\n  Running: LatencyMin DDPG ({num_episodes} episodes)...")
    infra, vehicles, edge_servers, cloud = build_infrastructure()
    mobility = make_mobility(vehicles, edge_servers)
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)

    all_delays, all_energies = [], []
    for ep in range(num_episodes):
        delays, energies = run_latencymin_ddpg_episode(
            vehicles, edge_servers, cloud, mobility, agent
        )
        mobility.current_slot = 0  # reset slot for next episode
        all_delays.append(sum(delays))
        all_energies.append(sum(energies))
        agent.decay_noise()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"    Ep {ep+1:4d}/{num_episodes} | Delay: {all_delays[-1]:.3f} s")

    agent.save(os.path.join(results_dir, "ddpg_latencymin.pth"))
    _save(results_dir, "LatencyMin", all_delays, all_energies)
    return all_delays, all_energies


# ─────────────────────────────────────────────────────────────────────────────
# Scheme 3: Energy Minimization via DDPG (reward = -energy only)
# ─────────────────────────────────────────────────────────────────────────────

def run_energymin(num_episodes: int, results_dir: str):
    print(f"\n  Running: EnergyMin DDPG ({num_episodes} episodes)...")
    infra, vehicles, edge_servers, cloud = build_infrastructure()
    mobility = make_mobility(vehicles, edge_servers)
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)

    all_delays, all_energies = [], []
    for ep in range(num_episodes):
        delays, energies = run_energymin_ddpg_episode(
            vehicles, edge_servers, cloud, mobility, agent
        )
        mobility.current_slot = 0
        all_delays.append(sum(delays))
        all_energies.append(sum(energies))
        agent.decay_noise()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"    Ep {ep+1:4d}/{num_episodes} | Delay: {all_delays[-1]:.3f} s")

    agent.save(os.path.join(results_dir, "ddpg_energymin.pth"))
    _save(results_dir, "EnergyMin", all_delays, all_energies)
    return all_delays, all_energies


# ─────────────────────────────────────────────────────────────────────────────
# Schemes 4, 5, 6: Heuristic baselines (Random, LRU, Cloud)
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic_baselines(num_episodes: int, results_dir: str):
    heuristics = {
        "NoCache": run_nocache_episode,
        "Random":  run_random_episode,
        "LRU":     run_lru_episode,
        "Cloud":   run_cloud_episode,
    }
    for name, fn in heuristics.items():
        print(f"\n  Running: {name} ({num_episodes} episodes)...")
        infra, vehicles, edge_servers, cloud = build_infrastructure()
        mobility = make_mobility(vehicles, edge_servers)
        all_delays, all_energies = [], []
        for ep in range(num_episodes):
            delays, energies = fn(vehicles, edge_servers, cloud, mobility)
            all_delays.append(sum(delays))
            all_energies.append(sum(energies))
            if (ep + 1) % 50 == 0 or ep == 0:
                print(f"    Ep {ep+1:4d}/{num_episodes} | Delay: {all_delays[-1]:.3f} s")
        _save(results_dir, name, all_delays, all_energies)


def _save(results_dir, name, delays, energies):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"metrics_{name}.json")
    with open(path, "w") as f:
        json.dump({"delays": delays, "energies": energies}, f)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VEC Caching LEAF Simulation")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help=f"Number of episodes (default: {NUM_EPISODES})")
    parser.add_argument("--scheme", type=str, default="all",
                        choices=["all", "ddpg", "baselines"],
                        help="Which scheme(s) to run")
    parser.add_argument("--results-dir", type=str,
                        default="examples/vec_caching/results",
                        help="Directory to save results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # DDPG-based schemes (3 agents, different reward signals)
    if args.scheme in ("all", "ddpg"):
        run_proposed(args.episodes, args.results_dir)
        run_latencymin(args.episodes, args.results_dir)
        run_energymin(args.episodes, args.results_dir)

    # Single heuristic baseline: NoCache (Baseline 1 from paper)
    if args.scheme in ("all", "baselines"):
        print(f"\n  Running: NoCache ({args.episodes} episodes)...")
        infra, vehicles, edge_servers, cloud = build_infrastructure()
        mobility = make_mobility(vehicles, edge_servers)
        all_delays, all_energies = [], []
        for ep in range(args.episodes):
            delays, energies = run_nocache_episode(vehicles, edge_servers, cloud, mobility)
            all_delays.append(sum(delays))
            all_energies.append(sum(energies))
            if (ep + 1) % 50 == 0 or ep == 0:
                print(f"    Ep {ep+1:4d}/{args.episodes} | Delay: {all_delays[-1]:.3f} s")
        _save(args.results_dir, "NoCache", all_delays, all_energies)

    print("\n  Done! Run plot_results.py to generate figures.")
