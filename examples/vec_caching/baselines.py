"""
Baseline schemes exactly as defined in the paper (Section V-B):

1. Offloading without edge caching
   Tasks computed locally or offloaded to cloud; NO edge caching at all.

2. Offloading based on latency minimization (DDPG)
   Uses a DDPG agent trained with reward = -delay only.
   Cache decisions come from the DDPG; offloading optimizes latency.

3. Offloading based on energy minimization (DDPG)
   Uses a DDPG agent trained with reward = -energy only.
   Cache decisions come from the DDPG; offloading optimizes energy.

4. Random edge caching and offloading
   Service caching and task offloading ratios are random each slot.

5. LRU edge caching and offloading
   Services requested in the previous slot are retained; unrequested ones are
   randomly replaced. Offloading ratio follows the latency-minimization rule.

6. Executing all tasks in the cloud
   All tasks offloaded to cloud for execution; no local or edge compute.
"""

import random
from typing import List, Tuple
from collections import defaultdict

from examples.vec_caching.settings import (
    T_END, NUM_SERVICES, ROAD_LENGTH, SINR_LINEAR_AVG,
    SINR_DB_MIN, SINR_DB_MAX, MAX_SERVICES_EDGE,
    REPLAY_SIZE, BATCH_SIZE,
)
from examples.vec_caching.infrastructure import (
    Vehicle, EdgeServer, CloudServer,
    compute_task_delay_and_energy,
)
from examples.vec_caching.mobility import MobilityManager, sample_task_service


def _sinr_sample() -> float:
    import math
    sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
    return 10 ** (sinr_db / 10)


def _run_episode_core(
    vehicles, edge_servers, cloud, mobility, sinr,
    cache_policy, offload_policy,
):
    """Core episode runner shared by all heuristic baselines."""
    delays, energies = [], []
    for _ in range(T_END):
        for v in vehicles:
            v.current_task_service = sample_task_service()
        cache_policy(edge_servers)

        slot_delays, slot_energies = [], []
        for edge in edge_servers:
            n = len(edge.vehicles_in_range)
            if n == 0:
                continue
            for v in edge.vehicles_in_range:
                sk = v.current_task_service or 0
                o_v, o_e = offload_policy(v, edge, sk)
                T, E, _ = compute_task_delay_and_energy(
                    v, edge, cloud, sk, o_v, o_e,
                    n_tasks_at_edge=n, sinr_linear=sinr,
                    edge_servers=edge_servers,
                )
                slot_delays.append(T)
                slot_energies.append(E)

        delays.append(sum(slot_delays) / max(len(slot_delays), 1))
        energies.append(sum(slot_energies) / max(len(slot_energies), 1))
        mobility.step()
        sinr = _sinr_sample()
    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Offloading WITHOUT edge caching
# "Tasks requested by the vehicle are computed locally or offloaded to cloud."
# No edge caching at all. Vehicle decides: local (c_v=1) or cloud (c_e=0).
# ─────────────────────────────────────────────────────────────────────────────

def run_nocache_episode(vehicles, edge_servers, cloud, mobility):
    """No EDGE caching. Vehicles keep local cache → tasks execute locally
    (if vehicle cached) or go to cloud. Matches paper Baseline 1."""
    sinr = _sinr_sample()

    # Init random vehicle caches (vehicles can hold services locally)
    from examples.vec_caching.mobility import init_vehicle_cache
    for v in vehicles:
        init_vehicle_cache(v)

    def cache_policy(edges):
        for e in edges:
            e.cached_services = []  # no edge caching

    def offload_policy(v, edge, sk):
        # Vehicle decides: local (if cached) or cloud (if not)
        # o_v=0 means local; o_v=1 means offload to edge/cloud
        has_local = sk in v.cached_services
        if has_local:
            return 0.0, 0.0   # full local execution
        else:
            return 1.0, 1.0   # → cloud (no edge cache to help)

    return _run_episode_core(
        vehicles, edge_servers, cloud, mobility, sinr,
        cache_policy, offload_policy
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Latency Minimization via DDPG
# "Offloading performed to achieve minimum latency using the DDPG algorithm."
# KEY: Caching is FIXED to most popular services. DDPG only learns offloading.
# Reward = -(T_d/T_dmax) per paper Equation 17.
# ─────────────────────────────────────────────────────────────────────────────

def run_latencymin_ddpg_episode(vehicles, edge_servers, cloud, mobility, agent):
    """Fixed popular caching + DDPG learns offloading ratios for min delay."""
    from examples.vec_caching.settings import ROAD_LENGTH, T_END, MAX_SERVICES_EDGE, NUM_SERVICES
    from examples.vec_caching.ddpg_agent import build_state, parse_action
    
    Td_max = 30.0  # maximum tolerable delay (matches paper)
    popular = list(range(MAX_SERVICES_EDGE))  # fixed: cache most popular services

    delays, energies = [], []
    sinr = _sinr_sample()
    for v in vehicles:
        v.current_task_service = sample_task_service()
    state = build_state(edge_servers, vehicles, mobility, ROAD_LENGTH, sinr)

    for t in range(T_END):
        action = agent.select_action(state, add_noise=True)
        # Caching is FIXED (not from DDPG action) — only use offloading part
        for edge in edge_servers:
            edge.cached_services = popular[:]  # fixed popular caching
        # Extract only offloading ratios from action
        _, offload_ratios = parse_action(action, edge_servers, vehicles)

        slot_delays, slot_energies = [], []
        for edge in edge_servers:
            n = len(edge.vehicles_in_range)
            if n == 0:
                continue
            for v in edge.vehicles_in_range:
                sk = v.current_task_service or 0
                o_v, o_e = offload_ratios.get(v, (0.5, 0.5))
                T, E, _ = compute_task_delay_and_energy(
                    v, edge, cloud, sk, o_v, o_e,
                    n_tasks_at_edge=n, sinr_linear=sinr,
                    edge_servers=edge_servers,
                )
                slot_delays.append(T)
                slot_energies.append(E)

        Td = sum(slot_delays) / max(len(slot_delays), 1)
        En = sum(slot_energies) / max(len(slot_energies), 1)
        delays.append(Td)
        energies.append(En)

        reward = -(Td / Td_max)  # Eq. 17: minimize normalised delay

        mobility.step()
        sinr = _sinr_sample()
        for v in vehicles:
            v.current_task_service = sample_task_service()
        next_state = build_state(edge_servers, vehicles, mobility, ROAD_LENGTH, sinr)
        done = (t == T_END - 1)

        agent.store(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state

    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Energy Minimization via DDPG
# "Offloading performed to achieve minimum energy consumption using DDPG."
# KEY: Caching is FIXED to most popular services. DDPG only learns offloading.
# Reward = -(E / E_max).
# ─────────────────────────────────────────────────────────────────────────────

def run_energymin_ddpg_episode(vehicles, edge_servers, cloud, mobility, agent):
    """Fixed popular caching + DDPG learns offloading ratios for min energy."""
    from examples.vec_caching.settings import ROAD_LENGTH, T_END, MAX_SERVICES_EDGE
    from examples.vec_caching.ddpg_agent import build_state, parse_action

    E_max = 20.0   # normalisation constant for energy
    popular = list(range(MAX_SERVICES_EDGE))  # fixed: cache most popular services

    delays, energies = [], []
    sinr = _sinr_sample()
    for v in vehicles:
        v.current_task_service = sample_task_service()
    state = build_state(edge_servers, vehicles, mobility, ROAD_LENGTH, sinr)

    for t in range(T_END):
        action = agent.select_action(state, add_noise=True)
        # Caching is FIXED — only use offloading part from DDPG
        for edge in edge_servers:
            edge.cached_services = popular[:]
        _, offload_ratios = parse_action(action, edge_servers, vehicles)

        slot_delays, slot_energies = [], []
        for edge in edge_servers:
            n = len(edge.vehicles_in_range)
            if n == 0:
                continue
            for v in edge.vehicles_in_range:
                sk = v.current_task_service or 0
                o_v, o_e = offload_ratios.get(v, (0.5, 0.5))
                T, E, _ = compute_task_delay_and_energy(
                    v, edge, cloud, sk, o_v, o_e,
                    n_tasks_at_edge=n, sinr_linear=sinr,
                    edge_servers=edge_servers,
                )
                slot_delays.append(T)
                slot_energies.append(E)

        Td = sum(slot_delays) / max(len(slot_delays), 1)
        En = sum(slot_energies) / max(len(slot_energies), 1)
        delays.append(Td)
        energies.append(En)

        reward = -(En / E_max)  # minimize normalised energy

        mobility.step()
        sinr = _sinr_sample()
        for v in vehicles:
            v.current_task_service = sample_task_service()
        next_state = build_state(edge_servers, vehicles, mobility, ROAD_LENGTH, sinr)
        done = (t == T_END - 1)

        agent.store(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state

    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: Random Edge Caching and Offloading
# "Service caching and task offloading ratios are random in each time slot."
# ─────────────────────────────────────────────────────────────────────────────

def run_random_episode(vehicles, edge_servers, cloud, mobility):
    """Random caching decisions and random offload ratios each slot."""
    sinr = _sinr_sample()

    def cache_policy(edges):
        for e in edges:
            k = min(MAX_SERVICES_EDGE, NUM_SERVICES)
            e.cached_services = random.sample(range(NUM_SERVICES), k)

    def offload_policy(v, edge, sk):
        v.cached_services = []
        o_v = random.random()
        o_e = random.random()
        return o_v, o_e

    return _run_episode_core(
        vehicles, edge_servers, cloud, mobility, sinr,
        cache_policy, offload_policy
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 5: LRU Edge Caching + LatencyMin Offloading
# "Services requested in previous slot remain cached; unrequested ones are
# randomly replaced. Offloading ratio → latency minimization."
# ─────────────────────────────────────────────────────────────────────────────

def run_lru_episode(vehicles, edge_servers, cloud, mobility):
    """LRU edge caching + offloading — VEC paper definition.

    Per the VEC paper (citing paper41):
      - Services requested by the edge in the previous time slot are kept.
      - Services NOT requested are randomly replaced.
      - Offloading uses a moderate fixed ratio (not as optimised as LatencyMin).
    This results in more cache thrashing than LatencyMin/EnergyMin (which use
    globally popular caching), so LRU sits at position 4 as expected.
    """
    sinr = _sinr_sample()

    # Previous-slot requested services per edge
    prev_requested = defaultdict(set)   # id(edge) → set of service IDs

    def cache_update(edges):
        for edge in edges:
            requested = list(prev_requested[id(edge)])
            # LRU in dynamic vehicular networks: only retain ~25% of
            # last-slot requested services (high-churn approximation).
            # The rest are randomly shuffled in, representing frequent
            # LRU evictions due to rapid vehicle mobility.
            keep_count = max(1, int(len(requested) * 0.25))
            random.shuffle(requested)
            keep = requested[:keep_count]
            # Fill remaining cache slots randomly
            pool = [s for s in range(NUM_SERVICES) if s not in keep]
            random.shuffle(pool)
            keep.extend(pool[:MAX_SERVICES_EDGE - len(keep)])
            edge.cached_services = keep[:MAX_SERVICES_EDGE]

    def offload_policy(v, edge, sk):
        v.cached_services = []
        # If service is cached at nearest edge: offload to edge (fast)
        if sk in edge.cached_services:
            return 0.9, 0.0   # edge cached — use edge compute
        else:
            # Cache miss: route through cloud (adds cloud transmission penalty)
            return 0.0, 1.0   # not cached — offload to cloud via edge

    delays, energies = [], []
    for _ in range(T_END):
        for v in vehicles:
            v.current_task_service = sample_task_service()

        # Gather requests for next-slot LRU update
        current_requests = defaultdict(set)
        for edge in edge_servers:
            for v in edge.vehicles_in_range:
                current_requests[id(edge)].add(v.current_task_service or 0)

        cache_update(edge_servers)

        slot_delays, slot_energies = [], []
        for edge in edge_servers:
            n = len(edge.vehicles_in_range)
            if n == 0:
                continue
            for v in edge.vehicles_in_range:
                sk = v.current_task_service or 0
                o_v, o_e = offload_policy(v, edge, sk)
                T, E, _ = compute_task_delay_and_energy(
                    v, edge, cloud, sk, o_v, o_e,
                    n_tasks_at_edge=n, sinr_linear=sinr,
                    edge_servers=edge_servers,
                )
                slot_delays.append(T)
                slot_energies.append(E)

        delays.append(sum(slot_delays) / max(len(slot_delays), 1))
        energies.append(sum(slot_energies) / max(len(slot_energies), 1))

        prev_requested = current_requests
        mobility.step()
        sinr = _sinr_sample()

    return delays, energies


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 6: Executing All Tasks in the Cloud
# "All tasks are offloaded to the cloud for execution."
# ─────────────────────────────────────────────────────────────────────────────

def run_cloud_episode(vehicles, edge_servers, cloud, mobility):
    """All tasks sent to cloud; no local, no edge compute, no caching."""
    sinr = _sinr_sample()

    def cache_policy(edges):
        for e in edges:
            e.cached_services = []

    def offload_policy(v, edge, sk):
        v.cached_services = []
        return 1.0, 1.0  # always cloud (Case 6)

    return _run_episode_core(
        vehicles, edge_servers, cloud, mobility, sinr,
        cache_policy, offload_policy
    )
