import numpy as np
from typing import Union  # ADDED: Import Union for type hinting
from core.parameters import *
from core.position import haversine
from core.model import Vehicle, Server, Task  # Import Task, Vehicle, Server for type hinting and checks


# Helper function to ensure consistency with simulator's transmission calculations
def _calculate_transmission_time_greedy(task: Task, distance: float, tx_power: float, bandwidth: float) -> float:
    """
    Calculates transmission time, consistent with simulator.py's model.
    """
    distance = max(distance, 0.001)  # 1m minimum

    # Convert TX_POWER from Watts to dBm
    tx_power_dbm = 10 * np.log10(tx_power * 1000)  # Convert Watts to mW, then to dBm

    # Path loss in dB (using log10_distance directly)
    log10_distance = np.log10(distance)
    path_loss_db = 38.02 + 20 * log10_distance

    # Received power in dBm
    received_power_dbm = tx_power_dbm - path_loss_db

    # Convert NOISE_POWER from Watts to dBm
    noise_power_dbm = 10 * np.log10(NOISE_POWER * 1000)

    # Received power and noise power in linear mW for SINR calculation
    received_power_mw = 10 ** (received_power_dbm / 10)
    noise_power_mw = 10 ** (noise_power_dbm / 10)

    sinr = received_power_mw / noise_power_mw
    sinr = max(min(sinr, 1e6), 1e-3)  # SINR must be > 0 for log2

    # Data rate using Shannon capacity formula (in bits/sec)
    data_rate = bandwidth * np.log2(1 + sinr)

    # Ensure minimum data rate
    data_rate = max(data_rate, 1000)  # 1 Kbps minimum

    # Transmission time with minimum threshold
    return max(task.size_bits / data_rate, 0.001)  # 1ms minimum


class GreedyOffloader:
    """
    Implements a greedy offloading strategy based on minimizing a weighted cost function
    (latency + energy). It considers Local, RSU, Base Station, and Cloud options.
    Includes specific RSU range check from the provided greedy.py.
    """

    def __init__(self, env, vehicle: Vehicle, task: Task, available_servers: list[Server]):
        self.env = env
        self.vehicle = vehicle
        self.task = task
        # Filter servers to only include RSU, BS, and Cloud as per the original greedy.py's find_best_node
        self.available_infrastructure_servers = [s for s in available_servers if
                                                 s.server_type in ['RSU', 'BS', 'Cloud']]

    def _calculate_cost(self, target_node: Union[Server, None], distance: float, task: Task) -> tuple[
        float, float, float, float, float]:
        """
        Calculates the weighted cost (latency + energy) for a given offloading target.
        Returns (cost, total_time, comm_time, new_deadline, proc_time).
        Returns inf for cost if not feasible.
        """
        # --- Local Processing Cost ---
        if target_node is None:  # Represents local processing
            local_processing_time = task.required_cycles / self.vehicle.frequency
            energy_consumed = ALPHA * task.required_cycles  # Local energy

            if local_processing_time > task.deadline:
                return float('inf'), float('inf'), 0.0, 0.0, 0.0  # Cost, total_time, comm_time, new_deadline, proc_time

            cost = (WEIGHTS[0] * local_processing_time) + (WEIGHTS[1] * energy_consumed)
            return cost, local_processing_time, 0.0, task.deadline, local_processing_time

        # --- Offloading to Server Cost ---
        # Communication time (uplink + downlink)
        comm_time = _calculate_transmission_time_greedy(task, distance, TX_POWER, BANDWIDTH_RANGE[0])

        # Energy consumed for transmission (from source vehicle)
        energy_consumed_tx = TX_POWER * comm_time  # Simplified, assuming only uplink energy for decision

        # New deadline after communication
        new_deadline = task.deadline - comm_time
        if new_deadline <= 0:
            return float('inf'), float('inf'), comm_time, new_deadline, 0.0

        # Processing time on the server
        processing_power_required = task.required_cycles / new_deadline  # Required power to meet new deadline

        # Ensure server has enough frequency and is not fully utilized
        available_freq = target_node.frequency * (1.0 - getattr(target_node, 'utilization', 0.0))
        if available_freq <= 0:  # Server is fully utilized or has no frequency
            return float('inf'), float('inf'), comm_time, new_deadline, 0.0

        # If required_cycles is 0, processing_time is 0, avoid division by zero for available_freq
        if task.required_cycles == 0:
            proc_time = 0.0
        else:
            proc_time = task.required_cycles / available_freq  # Actual processing time given available freq

        # Total time
        total_time = comm_time + proc_time + (comm_time * OP_IP_RATIO)  # Include result return time

        # Check if total time exceeds original deadline
        if total_time > task.deadline:
            return float('inf'), float('inf'), comm_time, new_deadline, proc_time

        # Total energy (transmission + processing if applicable, though greedy.py only considers tx energy for cost)
        # For consistency with greedy.py's cost, we'll use only transmission energy here for the cost calculation.
        # The simulator will track actual total energy.
        total_energy_for_cost = energy_consumed_tx

        cost = (WEIGHTS[0] * total_time) + (WEIGHTS[1] * total_energy_for_cost)

        return cost, total_time, comm_time, new_deadline, proc_time

    def solve(self):
        """
        Solves the offloading problem using a greedy approach.
        Returns the selected server object or None for local processing.
        """
        min_cost = float('inf')
        best_target = None

        # --- 1. Evaluate Local Processing First ---
        local_cost, local_total_time, _, _, _ = self._calculate_cost(None, 0.0, self.task)

        if local_cost < min_cost:
            min_cost = local_cost
            best_target = None  # Represents local processing

        # --- 2. Evaluate Infrastructure Servers (RSU, BS, Cloud) ---
        for server in self.available_infrastructure_servers:
            distance = haversine(*self.vehicle.position, *server.position)

            # Specific RSU range check from greedy.py
            if server.server_type == 'RSU':
                # The original greedy.py has a condition: if distance < 0.3 (km) and deadline > time_to_leave,
                # it forces local processing. We'll integrate this by making RSU cost infinite if this condition holds.
                if distance < 0.3:  # Assuming 0.3km is a critical close range
                    # Calculate time to leave RSU range, assuming linear movement directly away
                    # Convert RSU_RANGE_KM to meters, distance to meters, speed to m/s
                    if self.vehicle.speed > 0:
                        # If current distance is already greater than RSU_RANGE_KM, it's already out of range
                        if distance > RSU_RANGE_KM:
                            continue  # Skip this RSU

                        time_to_leave_rsu = (RSU_RANGE_KM - distance) / (self.vehicle.speed / 3.6)  # Convert km to km/s

                        if self.task.deadline > time_to_leave_rsu:
                            # Force local processing by making this RSU's cost infinite
                            # This mimics the greedy.py's behavior of returning None (local)
                            # if this RSU condition is met. We will skip this RSU as a candidate.
                            continue

                            # Calculate cost for offloading to this server
            cost, total_time, comm_time, new_deadline, proc_time = self._calculate_cost(server, distance, self.task)

            # Check if this server is a better option
            if cost < min_cost:
                min_cost = cost
                best_target = server

        # If the best option found is still infinite, it means no feasible option
        if min_cost == float('inf'):
            return None  # Fallback to local (or signify failure if local also inf)

        return best_target
