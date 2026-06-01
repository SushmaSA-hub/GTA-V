import numpy as np
import math
from core.parameters import *  # Import global simulation parameters
from core.position import haversine  # For distance calculations
from core.model import Vehicle, Server, Task  # For type hinting and accessing attributes


# This function remains the same as it's a generic transmission time calculation
def calculate_transmission_time_for_deco(task: Task, distance: float, tx_power: float, bandwidth: float) -> float:
    """
    Calculates transmission time, adapted from simulator.py for DECO's use.
    Uses specific TX_POWER and BANDWIDTH.
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


class DECOOffloader:
    """
    Implements the DECO (Deadline-aware and Energy-Efficient Computation Offloading) algorithm
    as a baseline, adapted from the provided paper.
    This version considers only local processing or offloading to RSU/Base Station (edge servers).
    V2V and Cloud offloading are explicitly excluded for this baseline.
    """

    # CORRECTED: Removed all_vehicles from the constructor signature
    def __init__(self, env, vehicle: Vehicle, task: Task, available_servers: list[Server]):
        self.env = env
        self.vehicle = vehicle
        self.task = task
        # Filter available_servers to only include RSUs and Base Stations for DECO baseline
        # Cloud is also excluded as per the original DECO paper's focus on edge.
        self.available_edge_servers = [s for s in available_servers if s.server_type in ['RSU', 'BS']]
        # self.all_vehicles is no longer part of this class's state or used in this version

    def _calculate_local_time(self) -> float:
        """Calculates the time required to compute a task locally on an IoT device."""
        if self.vehicle.frequency == 0:
            return float('inf')
        return self.task.required_cycles / self.vehicle.frequency

    def _calculate_local_energy(self) -> float:
        """
        Calculates the energy consumed for local processing using DECO's formula.
        """
        #  return ALPHA * (self.vehicle.frequency ** 2) * self.task.required_cycles
        return ALPHA * self.task.required_cycles

    def _calculate_offload_transmission_metrics(self, target_server: Server) -> tuple[float, float]:
        """
        Calculates transmission delay and energy for offloading to an infrastructure server (RSU, BS).
        Includes uplink and downlink.
        """
        distance = haversine(*self.vehicle.position, *target_server.position)

        # Uplink transmission time for input data
        uplink_tx_time = calculate_transmission_time_for_deco(
            Task(self.task.vehicle_id, self.task.size_bits, self.task.deadline, self.task.created_time),
            distance, TX_POWER, BANDWIDTH_RANGE[0]
        )

        # Downlink transmission time for output data (using OP_IP_RATIO)
        output_data_size_bits = self.task.size_bits * OP_IP_RATIO
        dummy_output_task = Task(self.task.vehicle_id, output_data_size_bits, self.task.deadline,
                                 self.task.created_time)
        downlink_tx_time = calculate_transmission_time_for_deco(
            dummy_output_task, distance, TX_POWER, BANDWIDTH_RANGE[0]  # Assuming symmetric comms
        )

        total_tx_delay = uplink_tx_time + downlink_tx_time

        # Energy consumed for transmission (only uplink from device)
        tx_energy = TX_POWER * uplink_tx_time  # Paper's formula: rho_k^tx * tau_k,l^tx

        return total_tx_delay, tx_energy

    def _calculate_processing_delay_ecs(self, server: Server) -> float:
        """Calculates the time taken for processing a task at an ECS."""
        return self.task.required_cycles / server.frequency

    def _estimate_queuing_delay(self, server: Server, processing_time: float) -> float:
        """
        Simplistic estimation of queuing delay for DECO baseline.
        For this baseline, we return 0. The actual queuing is handled by SimPy's resource.
        """
        return 0.0

    # Removed _find_eligible_v2v_candidates_for_deco and _calculate_v2v_offload_potential_for_deco

    def solve(self):
        """
        Executes the DECO algorithm's decision logic, considering only local or edge server offloading.
        Returns the selected server object (RSU/BS) or None for local processing.
        """
        # --- 1. Local Decision Phase (Algorithm 1 from DECO paper) ---
        tau_k_loc = self._calculate_local_time()
        epsilon_k_loc = self._calculate_local_energy()

        # Find the nearest RSU or BS for comparison in Algorithm 1
        nearest_infrastructure_server = None
        min_dist_to_infra = float('inf')

        for server in self.available_edge_servers:  # Only iterate through edge servers
            distance = haversine(*self.vehicle.position, *server.position)
            if distance < min_dist_to_infra:
                min_dist_to_infra = distance
                nearest_infrastructure_server = server

        tau_k_l_tx = float('inf')  # Transmission delay for input to nearest infra server
        epsilon_k_tx = float('inf')  # Transmission energy for input to nearest infra server

        if nearest_infrastructure_server:
            # For Algorithm 1, consider only input data transmission to nearest BS/RSU
            input_data_task = Task(self.task.vehicle_id, self.task.size_bits, self.task.deadline,
                                   self.task.created_time)
            tau_k_l_tx = calculate_transmission_time_for_deco(
                input_data_task, min_dist_to_infra, TX_POWER, BANDWIDTH_RANGE[0]
            )
            epsilon_k_tx = TX_POWER * tau_k_l_tx

        # Apply DECO Local Decision Logic (Algorithm 1)
        decision = None

        # Case 1: Local processing is better for both delay and energy
        if tau_k_loc <= tau_k_l_tx and epsilon_k_loc <= epsilon_k_tx:
            decision = 'local'
        # Case 2: Offloading is better for both delay and energy
        elif tau_k_l_tx <= tau_k_loc and epsilon_k_tx <= epsilon_k_loc:
            decision = 'offload'
        # Case 3: Local for delay, offloading for energy. Check deadline for offloading.
        elif tau_k_loc < tau_k_l_tx and epsilon_k_tx < epsilon_k_loc:
            if tau_k_l_tx <= self.task.deadline:  # If offloading (transmission only) meets deadline
                decision = 'offload'
            else:  # Offloading (transmission only) violates deadline, so process locally
                decision = 'local'
        # Case 4: Offloading for delay, local for energy. Check deadline for local.
        elif tau_k_l_tx < tau_k_loc and epsilon_k_loc < epsilon_k_tx:
            if tau_k_loc <= self.task.deadline:  # If local processing meets deadline
                decision = 'local'
            else:  # Local processing violates deadline, so offload
                decision = 'offload'
        else:  # Fallback: if no clear best case, default to local
            decision = 'local'

        # If local decision is to process locally, return None
        if decision == 'local':
            # If local processing itself exceeds deadline, it's a failure.
            # DECO Algorithm 1 pushes to offload if local fails deadline, so if we reach here
            # with 'local' decision, it means local is the best viable option.
            if tau_k_loc > self.task.deadline:
                return None  # Task will fail locally, so no viable option
            return None  # Process locally

        # --- 2. Scheduling Offloaded Tasks Phase (Algorithm 2 from DECO paper - Edge Servers Only) ---
        # If the decision is to offload, find the best target among available edge servers.

        min_total_offload_delay = float('inf')
        best_target = None

        # Consider only edge servers (RSU, BS)
        for server in self.available_edge_servers:
            # Check RSU mobility stability (similar to your proposed model)
            if server.server_type == 'RSU':
                distance_to_rsu = haversine(*self.vehicle.position, *server.position)
                if self.vehicle.speed > 0:
                    time_to_leave_rsu = (RSU_RANGE_KM * 1000 - (distance_to_rsu * 1000)) / (self.vehicle.speed / 3.6)
                    if time_to_leave_rsu < self.task.deadline:
                        continue  # Skip this RSU if vehicle won't stay in range

            # Calculate transmission delay (uplink + downlink)
            current_tx_delay, _ = self._calculate_offload_transmission_metrics(server)

            # Calculate processing delay on this ECS
            current_proc_delay = self._calculate_processing_delay_ecs(server)

            # Estimate queuing delay (simplified to 0 for this baseline)
            current_queuing_delay = self._estimate_queuing_delay(server, current_proc_delay)

            # Total offloading delay (Equation 7 from paper)
            current_offload_delay = current_tx_delay + current_proc_delay + current_queuing_delay

            # Check against task deadline
            if current_offload_delay > self.task.deadline:
                continue  # This server cannot meet the deadline

            if current_offload_delay < min_total_offload_delay:
                min_total_offload_delay = current_offload_delay
                best_target = server

        return best_target  # Returns the best edge server or None if no suitable offload target found
