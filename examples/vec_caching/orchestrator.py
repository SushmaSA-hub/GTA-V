"""
DDPG-based orchestrator for joint service caching and computation offloading.

Extends LEAF's Orchestrator abstract class, using the DDPG agent to:
1. Decide which services each edge server should cache
2. Decide what offloading ratio each vehicle uses

The `run_episode()` method executes one full episode (T_END time slots),
returning per-slot delay and energy metrics.
"""

import random
import math
from typing import List, Dict, Tuple

import numpy as np

from leaf.infrastructure import Infrastructure
from leaf.orchestrator import Orchestrator
from leaf.application import Application, SourceTask, ProcessingTask, SinkTask

from examples.vec_caching.settings import (
    T_END, NUM_VEHICLES, NUM_EDGES, NUM_SERVICES,
    ROAD_LENGTH, SINR_LINEAR_AVG, SINR_DB_MIN, SINR_DB_MAX,
)
from examples.vec_caching.infrastructure import (
    Vehicle, EdgeServer, CloudServer,
    compute_task_delay_and_energy,
)
from examples.vec_caching.mobility import MobilityManager, sample_task_service
from examples.vec_caching.ddpg_agent import DDPGAgent, build_state, parse_action


def _sinr_sample() -> float:
    sinr_db = random.uniform(SINR_DB_MIN, SINR_DB_MAX)
    return 10 ** (sinr_db / 10)


class VECOrchestrator(Orchestrator):
    """
    Orchestrates VEC tasks using a DDPG agent.
    The LEAF Orchestrator API is used for task placement tracking;
    actual delay/energy are computed analytically via the paper's equations.
    """

    def __init__(self,
                 infrastructure: Infrastructure,
                 vehicles: List[Vehicle],
                 edge_servers: List[EdgeServer],
                 cloud: CloudServer,
                 mobility_manager: MobilityManager,
                 agent: DDPGAgent,
                 training: bool = True,
                 warm_start_cache: bool = False):
        super().__init__(infrastructure)
        self.vehicles = vehicles
        self.edge_servers = edge_servers
        self.cloud = cloud
        self.mobility = mobility_manager
        self.agent = agent
        self.training = training
        self.warm_start_cache = warm_start_cache

    # ── Public API ────────────────────────────────────────────────────────────

    def run_episode(self) -> Tuple[List[float], List[float]]:
        """
        Execute one episode over T_END time slots.

        Returns:
            delays: list of Td(t) for t=1..T_END
            energies: list of avg energy for t=1..T_END
        """
        delays = []
        energies = []

        # Warm-start: pre-load edge caches with popular services so the DDPG
        # starts from a competitive baseline (same as LatencyMin/EnergyMin).
        if self.warm_start_cache:
            from examples.vec_caching.settings import MAX_SERVICES_EDGE
            popular = list(range(MAX_SERVICES_EDGE))
            for edge in self.edge_servers:
                edge.cached_services = popular[:]

        # Sample initial state
        sinr = _sinr_sample()
        for v in self.vehicles:
            v.current_task_service = sample_task_service()
        state = build_state(
            self.edge_servers, self.vehicles,
            self.mobility, ROAD_LENGTH, sinr
        )

        for t in range(T_END):
            # Agent selects action
            action = self.agent.select_action(state, add_noise=self.training)

            # Apply caching decisions
            edge_cache, offload_ratios = parse_action(
                action, self.edge_servers, self.vehicles
            )
            for edge, services in edge_cache.items():
                edge.cached_services = services

            # Execute tasks — compute delay/energy for each vehicle
            slot_delays = []
            slot_energies = []

            for edge in self.edge_servers:
                n_tasks = len(edge.vehicles_in_range)
                if n_tasks == 0:
                    continue
                for v in edge.vehicles_in_range:
                    service_k = v.current_task_service or 0
                    o_v, o_e = offload_ratios.get(v, (0.5, 0.5))

                    T_total, E_total, _ = compute_task_delay_and_energy(
                        v, edge, self.cloud,
                        service_k, o_v, o_e,
                        n_tasks_at_edge=n_tasks,
                        sinr_linear=sinr,
                        edge_servers=self.edge_servers,
                    )
                    slot_delays.append(T_total)
                    slot_energies.append(E_total)

            # Average delay/energy this slot (Eq 15)
            Td_t = sum(slot_delays) / max(len(slot_delays), 1)
            E_t = sum(slot_energies) / max(len(slot_energies), 1)
            delays.append(Td_t)
            energies.append(E_t)

            # Reward = negative normalised delay (Eq. 17 objective)
            reward = -(Td_t / 30.0)  # T_dmax = 30s

            # Step mobility, reassign tasks
            self.mobility.step()
            sinr = _sinr_sample()
            for v in self.vehicles:
                v.current_task_service = sample_task_service()

            next_state = build_state(
                self.edge_servers, self.vehicles,
                self.mobility, ROAD_LENGTH, sinr
            )
            done = (t == T_END - 1)

            if self.training:
                self.agent.store(state, action, reward, next_state, done)
                self.agent.train_step()

            state = next_state

        return delays, energies

    # ── Required abstract method (LEAF) ───────────────────────────────────────

    def _processing_task_placement(self, processing_task, application):
        """Place on cloud by default (LEAF routing placeholder)."""
        return self.infrastructure.nodes(type_filter=CloudServer)[0]
