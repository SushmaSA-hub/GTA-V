"""
Vehicle mobility model for the VEC caching simulation.

Vehicles move on a 1-D road of length = NUM_EDGES × EDGE_COMM_RANGE.
Edge servers are placed at the centre of each segment.
At each time slot vehicles move and update their associated edge server.
"""

import random
from typing import List, Optional

from examples.vec_caching.settings import (
    V_MIN, V_MAX, DELTA_T, ROAD_LENGTH,
    NUM_EDGES, EDGE_COMM_RANGE, NUM_SERVICES, ZIPF_EXPONENT,
    USE_SUMO, SUMO_TRACE_FILE
)
from examples.vec_caching.infrastructure import Vehicle, EdgeServer
from examples.vec_caching.sumo_utils import load_sumo_trace, get_edge_hotspots


class MobilityManager:
    """Manages vehicle positions and edge associations per time slot."""

    def __init__(self, vehicles: List[Vehicle], edge_servers: List[EdgeServer]):
        self.vehicles = vehicles
        self.edge_servers = edge_servers
        # Map: vehicle → nearest edge
        self.vehicle_edge_map: dict = {}
        self._init_positions()
        self._update_associations()

    def _init_positions(self):
        """Randomly initialise vehicle positions and speeds on the road."""
        for v in self.vehicles:
            v.position = random.uniform(0, ROAD_LENGTH)
            v.speed = random.uniform(V_MIN, V_MAX)

    def step(self):
        """Advance one time slot: move vehicles and update edge associations."""
        for v in self.vehicles:
            # Move forward by speed × Δt, wrap around road
            v.position = (v.position + v.speed * DELTA_T) % ROAD_LENGTH
            # Randomly change speed each slot
            v.speed = random.uniform(V_MIN, V_MAX)
        self._update_associations()

    def _update_associations(self):
        """Assign each vehicle to its nearest edge server."""
        for edge in self.edge_servers:
            edge.vehicles_in_range = []

        for v in self.vehicles:
            nearest = min(self.edge_servers, key=lambda e: abs(v.position - e.position))
            self.vehicle_edge_map[v] = nearest
            nearest.vehicles_in_range.append(v)

    def get_edge(self, vehicle: Vehicle) -> EdgeServer:
        return self.vehicle_edge_map.get(vehicle, self.edge_servers[0])

    def get_vehicles_at_edge(self, edge: EdgeServer) -> List[Vehicle]:
        return edge.vehicles_in_range


class SumoMobilityManager:
    """Manages vehicle positions from a SUMO 2D macro-mobility trace."""

    def __init__(self, vehicles: List[Vehicle], edge_servers: List[EdgeServer]):
        print(f"Loading SUMO trace from {SUMO_TRACE_FILE}...")
        self.trace_data = load_sumo_trace(SUMO_TRACE_FILE, delta_t=DELTA_T)
        print(f"Found traffic data for {len(self.trace_data)} simulation slots.")
        
        self.vehicles = vehicles
        self.edge_servers = edge_servers
        self.vehicle_edge_map: dict = {}
        self.current_slot = 0
        
        # Deploy edge servers to optimal hotspots
        hotspots = get_edge_hotspots(self.trace_data, num_edges=len(self.edge_servers))
        for i, edge in enumerate(self.edge_servers):
            edge.x, edge.y = hotspots[i]
            print(f"Deployed {edge.name} at (x={edge.x:.2f}, y={edge.y:.2f})")
            
        self._update_from_trace()
        self._update_associations()

    def step(self):
        """Advance one time slot: read next slot from trace."""
        self.current_slot += 1
        self._update_from_trace()
        self._update_associations()

    def _update_from_trace(self):
        """Update vehicle positions based on current trace data slot."""
        if not self.trace_data:
            return
            
        max_slot = max(self.trace_data.keys())
        active_slot = self.current_slot % (max_slot + 1)
        
        if active_slot not in self.trace_data:
            return
            
        slot_data = self.trace_data[active_slot]
        
        # The trace has vehicle IDs like 'veh0', 'veh1'. Our vehicles are just a pool.
        # So we map trace vehicles to our internal LEAF vehicle objects.
        # For simplicity, we assign the first N active trace vehicles to our N LEAF vehicles.
        active_coords = list(slot_data.values())
        
        # If there are fewer trace vehicles than our simulated pool, the remaining
        # pool vehicles are "parked" or keep their off-screen position.
        for i, v in enumerate(self.vehicles):
            if i < len(active_coords):
                x, y, speed = active_coords[i]
                v.x = x
                v.y = y
                v.speed = speed
            else:
                # Vehicle is not currently active in SUMO
                v.speed = 0.0

    def _update_associations(self):
        """Assign each vehicle to its nearest edge server using 2D Euclidean distance."""
        for edge in self.edge_servers:
            edge.vehicles_in_range = []

        import math
        for v in self.vehicles:
            nearest = min(self.edge_servers, key=lambda e: math.sqrt((v.x - e.x)**2 + (v.y - e.y)**2))
            self.vehicle_edge_map[v] = nearest
            nearest.vehicles_in_range.append(v)

    def get_edge(self, vehicle: Vehicle) -> EdgeServer:
        return self.vehicle_edge_map.get(vehicle, self.edge_servers[0])

    def get_vehicles_at_edge(self, edge: EdgeServer) -> List[Vehicle]:
        return edge.vehicles_in_range


def sample_task_service(num_services: int = NUM_SERVICES,
                        zipf_exp: float = ZIPF_EXPONENT) -> int:
    """Sample a service type according to Zipf distribution."""
    weights = [1.0 / (k ** zipf_exp) for k in range(1, num_services + 1)]
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(range(num_services), weights=weights)[0]


def init_vehicle_cache(vehicle: Vehicle, num_services: int = NUM_SERVICES):
    """Randomly initialise a vehicle's local service cache respecting storage limit."""
    from examples.vec_caching.settings import MAX_SERVICES_VEHICLE
    vehicle.cached_services = random.sample(
        range(num_services), min(MAX_SERVICES_VEHICLE, num_services)
    )


def init_edge_cache(edge: EdgeServer, num_services: int = NUM_SERVICES):
    """Randomly initialise an edge server's service cache respecting storage limit."""
    from examples.vec_caching.settings import MAX_SERVICES_EDGE
    edge.cached_services = random.sample(
        range(num_services), min(MAX_SERVICES_EDGE, num_services)
    )
