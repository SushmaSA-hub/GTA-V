"""
Infrastructure components for the VEC caching simulation.

Maps paper entities to LEAF nodes/links:
  - Vehicle  → leaf.infrastructure.Node (mobile, has CPU + local cache)
  - EdgeServer → leaf.infrastructure.Node (fixed, has CPU + service cache)
  - CloudServer → leaf.infrastructure.Node (unlimited compute)
  - LinkV2I    → leaf.infrastructure.Link (vehicle↔edge wireless)
  - LinkEdgeEdge → leaf.infrastructure.Link (edge↔edge wired)
  - LinkEdgeCloud → leaf.infrastructure.Link (edge↔cloud WAN)
"""

import math
import random
from typing import List, Optional

from leaf.infrastructure import Node, Link
from leaf.power import PowerModelNode, PowerModelLink

from examples.vec_caching.settings import (
    BANDWIDTH, SINR_LINEAR_AVG, DATA_SIZE,
    EDGE_CU, VEHICLE_CU, CLOUD_CU,
    EDGE_MAX_POWER, EDGE_STATIC_POWER,
    R_EDGE, R_CLOUD, TX_POWER_EDGE, TX_POWER_CLOUD,
    KAPPA, ALPHA, CPU_FREQ_EDGE, CPU_FREQ_VEHICLE,
    NUM_SERVICES, SERVICE_SIZE, STORAGE_VEHICLE, STORAGE_EDGE,
    MAX_SERVICES_VEHICLE, MAX_SERVICES_EDGE,
)

# ── Global counters for unique node names ─────────────────────────────────────
_vehicles_created = 0
_edges_created = 0


# ─────────────────────────────────────────────────────────────────────────────
# Compute Nodes
# ─────────────────────────────────────────────────────────────────────────────

class Vehicle(Node):
    """A mobile vehicle that generates computation tasks."""

    def __init__(self, position: float):
        global _vehicles_created
        name = f"vehicle_{_vehicles_created}"
        _vehicles_created += 1
        # Vehicles have limited CPU but no fixed power model (tracked analytically)
        super().__init__(name, cu=VEHICLE_CU, power_model=PowerModelNode(
            power_per_cu=KAPPA * (CPU_FREQ_VEHICLE ** (ALPHA - 1)),
        ))
        self.position = position          # position on the 1-D road (m)
        self.x = 0.0                      # 2D x coordinate
        self.y = 0.0                      # 2D y coordinate
        self.speed = 0.0                  # set by mobility manager

        # Service cache state: which of the NK services are cached here
        self.cached_services: List[int] = []  # list of service indices (0-based)

        # Task
        self.current_task_service: Optional[int] = None  # service type of current task


class EdgeServer(Node):
    """A fixed edge server (RSU / base station) with limited cache."""

    def __init__(self, position: float):
        global _edges_created
        name = f"edge_{_edges_created}"
        _edges_created += 1
        super().__init__(
            name,
            cu=EDGE_CU,
            power_model=PowerModelNode(
                max_power=EDGE_MAX_POWER,
                static_power=EDGE_STATIC_POWER,
            ),
        )
        self.position = position          # fixed position on road (m)
        self.x = 0.0
        self.y = 0.0
        self.comm_range = 500.0           # metres (Table III)

        # Service cache state
        self.cached_services: List[int] = []  # list of service indices

        # Vehicles currently in range (populated by mobility manager)
        self.vehicles_in_range: List[Vehicle] = []

    def is_vehicle_in_range(self, vehicle: Vehicle) -> bool:
        from examples.vec_caching.settings import USE_SUMO
        if USE_SUMO:
            return math.sqrt((vehicle.x - self.x)**2 + (vehicle.y - self.y)**2) <= self.comm_range / 2
        else:
            return abs(vehicle.position - self.position) <= self.comm_range / 2


class CloudServer(Node):
    """Cloud server with unlimited compute and storage."""

    def __init__(self):
        super().__init__("cloud", cu=None, power_model=PowerModelNode(power_per_cu=0))
        # Cloud always has all services cached
        self.cached_services: List[int] = list(range(NUM_SERVICES))


# ─────────────────────────────────────────────────────────────────────────────
# Network Links
# ─────────────────────────────────────────────────────────────────────────────

class LinkV2I(Link):
    """Wireless uplink from vehicle to nearest edge server.

    Bandwidth is shared equally among all vehicles in range of the same edge.
    Rate is computed dynamically by the orchestrator using Shannon's formula.
    We set a fixed average rate here for LEAF routing; actual rate computed analytically.
    """

    def __init__(self, vehicle: Vehicle, edge: EdgeServer):
        # Estimated average uplink rate (used for LEAF routing)
        rate = BANDWIDTH * math.log2(1 + SINR_LINEAR_AVG)
        super().__init__(
            vehicle, edge,
            bandwidth=rate,
            latency=0,
            power_model=PowerModelLink(energy_per_bit=0),  # power tracked analytically
        )


class LinkEdgeEdge(Link):
    """Wired link between edge servers (for cooperative offloading)."""

    def __init__(self, src: EdgeServer, dst: EdgeServer):
        super().__init__(
            src, dst,
            bandwidth=R_EDGE,
            latency=0,
            power_model=PowerModelLink(energy_per_bit=TX_POWER_EDGE / R_EDGE),
        )


class LinkEdgeCloud(Link):
    """WAN link from edge server to cloud."""

    def __init__(self, edge: EdgeServer, cloud: CloudServer):
        super().__init__(
            edge, cloud,
            bandwidth=R_CLOUD,
            latency=0,
            power_model=PowerModelLink(energy_per_bit=TX_POWER_CLOUD / R_CLOUD),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Delay & Energy Computation (Equations 5–16 of the paper)  
# ─────────────────────────────────────────────────────────────────────────────

def compute_transmission_rate(bandwidth_per_vehicle: float, sinr_linear: float) -> float:
    """Shannon's formula (Eq 2)."""
    return bandwidth_per_vehicle * math.log2(1 + sinr_linear)


def compute_task_delay_and_energy(
    vehicle: Vehicle,
    edge: EdgeServer,
    cloud: CloudServer,
    service_k: int,
    o_v: float,      # offload ratio from vehicle to nearest edge
    o_e: float,      # offload ratio from nearest edge to edge pool
    n_tasks_at_edge: int,  # Ne^task(t): number of tasks at this edge this slot
    sinr_linear: float = SINR_LINEAR_AVG,
    edge_servers: List[EdgeServer] = None, # to check edge pool
):
    """
    Compute total delay and energy for **one task** with offloading ratios (o_v, o_e).

    Returns (total_delay, total_energy) in seconds and joules.

    Implements Equations 7–16 and Table II (6 caching cases).
    """
    from examples.vec_caching.settings import (
        LAMBDA, DATA_SIZE, CPU_FREQ_VEHICLE, CPU_FREQ_EDGE,
        R_EDGE, R_CLOUD, KAPPA, ALPHA
    )

    dk = DATA_SIZE                   # data size (bits)
    lam = LAMBDA                     # cycles/bit
    fV = CPU_FREQ_VEHICLE            # cycles/s
    fE = CPU_FREQ_EDGE               # cycles/s

    # Bandwidth per vehicle (equally shared)
    bw = BANDWIDTH / max(n_tasks_at_edge, 1)
    R_V2E = compute_transmission_rate(bw, sinr_linear)

    # Determine caching case from Table II
    c_v = 1 if service_k in vehicle.cached_services else 0
    c_e = 1 if service_k in edge.cached_services else 0
    
    # Check if ANY other edge in the pool has the service
    if edge_servers is not None:
        c_pool = 1 if any(service_k in e.cached_services for e in edge_servers if e != edge) else 0
    else:
        c_pool = 0

    def phi(x):
        return 1 if x > 0 else 0

    # ── Determine case ────────────────────────────────────────────────────────
    if c_v == 1 and c_e == 0:
        # Case 1 or Case 2 (c_e=0 → nearest edge has no service)
        # If vehicle has service: nothing offloaded to edge (Case 1)
        o_v = 0.0
        case = 1
    elif c_v == 1 and c_e == 1:
        # Case 2: both vehicle and nearest edge have service
        # partial offloading to nearest edge only
        o_e = 0.0
        case = 2
    elif c_v == 0 and c_e == 1 and c_pool == 0:
        # Case 3: only nearest edge has it; no edge pool
        o_v = 1.0
        o_e = 0.0
        case = 3
    elif c_v == 0 and c_e == 1 and c_pool == 1:
        # Case 4: nearest edge + pool both have service
        o_v = 1.0
        case = 4
    elif c_v == 0 and c_e == 0 and c_pool == 1:
        # Case 5: only edge pool has service
        o_v = 1.0
        o_e = 1.0
        case = 5
    else:
        # Case 6: nobody has service → send to cloud
        o_v = 1.0
        o_e = 1.0
        case = 6

    # ── Local execution delay (Eq 7) ──────────────────────────────────────────
    if case == 1:
        T_local = lam * dk / fV                # full local execution
    else:
        T_local = c_v * (1 - o_v) * lam * dk / fV

    # ── Upload delay to nearest edge (Eq 8) ───────────────────────────────────
    if R_V2E > 0 and o_v > 0:
        # T_up is delay from vehicle to edge. Happens ANY time o_v > 0.
        T_up = o_v * dk / R_V2E
    else:
        T_up = 0.0

    # ── Nearest edge execution (Eq 10) ───────────────────────────────────────
    T_edge_exec = c_e * o_v * (1 - o_e) * lam * dk / fE

    # ── Edge pool upload delay (Eq 11) ────────────────────────────────────────
    if case in (4, 5):
        T_up_pool = (1 - c_v) * phi(c_pool) * o_v * o_e * dk / R_EDGE
    else:
        T_up_pool = 0.0

    # ── Edge pool execution (Eq 12) ──────────────────────────────────────────
    if case in (4, 5):
        T_pool_exec = (1 - c_v) * phi(c_pool) * o_v * o_e * lam * dk / fE
    else:
        T_pool_exec = 0.0

    # ── Cloud delay (Eq 16, last term) ────────────────────────────────────────
    if case == 6:
        # cloud: edge to cloud + download service
        # (V2E delay is already handled by T_up)
        T_cloud = (1 - c_v) * (1 - phi(c_e) * 1) * (dk / R_CLOUD + SERVICE_SIZE / R_CLOUD)
    else:
        T_cloud = 0.0

    # ── Total delay (Eq 16) ──────────────────────────────────────────────────
    T_total = (
        max(T_local, T_up)            # parallel: local + upload to edge
        + max(T_edge_exec, T_up_pool) # parallel: edge exec + upload to pool
        + T_pool_exec                 # pool execution
        + T_cloud                     # cloud path delay
    )

    # ── Energy Consumption (Eq 14) ───────────────────────────────────────────
    E_edge_compute = KAPPA * (fE ** ALPHA) * (T_edge_exec + T_pool_exec)
    E_tx_pool = TX_POWER_EDGE * T_up_pool
    E_tx_cloud = TX_POWER_CLOUD * (dk / R_CLOUD) if case == 6 else 0.0
    E_total = E_edge_compute + E_tx_pool + E_tx_cloud

    return T_total, E_total, case
