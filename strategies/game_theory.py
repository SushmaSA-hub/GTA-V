import numpy as np
from scipy.special import softmax
from core.parameters import *
from core.position import haversine
from core.model import Vehicle, Server, Task  # Import Task, Vehicle, Server for type hinting and checks


class PotentialGameOffloader:
    """
    Game-theoretic offloading: Considers Local, V2V, RSU, BS, and Cloud options.
    The decision is based on minimizing a potential function (cost).
    """

    def __init__(self, env, vehicle: Vehicle, task: Task, available_servers: list[Server], all_vehicles: list[Vehicle]):
        self.env = env
        self.vehicle = vehicle
        self.task = task
        self.available_servers = available_servers  # RSU, BS, Cloud that are in general range
        self.all_vehicles = all_vehicles  # All other vehicles in the simulation
        self.temperature = 1.0  # Controls the "rationality" of the decision-making (higher temp = more random)

    def _calculate_local_potential(self) -> float:
        """Calculate the potential (cost) for local processing."""
        computation_time = self.task.required_cycles / self.vehicle.frequency
        energy_consumption = ALPHA * self.task.required_cycles

        # If local processing exceeds deadline, assign very high potential
        if computation_time > self.task.deadline:
            return float('inf')

        # Combine time and energy (using weights if desired, but for simplicity, sum)
        return (computation_time * WEIGHTS[0]) + (energy_consumption * WEIGHTS[1])

    def _calculate_offload_potential(self, server: Server) -> float:
        """Calculate the potential (cost) for offloading to an RSU, BS, or Cloud server."""
        distance = haversine(self.vehicle.position[0], self.vehicle.position[1],
                             server.position[0], server.position[1])

        # Ensure minimum distance to prevent log(0) errors
        distance = max(distance, 0.001)  # 1m minimum

        # Convert TX_POWER from Watts to dBm for calculations
        tx_power_dbm = 10 * np.log10(TX_POWER * 1000)  # Convert Watts to mW, then to dBm

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

        # Simplified interference model: assume some base interference from other users
        # For a more complex model, you'd need to know current channel usage.
        server_load = getattr(server, 'utilization', 0.0)  # Utilization of the server
        # Estimate concurrent users based on server's queue or utilization
        concurrent_users = max(1, len(getattr(server, 'queue', [])) + 1)  # At least 1 user (this vehicle)

        # A simple interference model: interference increases with more users
        # This is a placeholder; a proper model would be more complex (e.g., based on channel allocation)
        interference_mw = (concurrent_users - 1) * (TX_POWER * 1000 * 0.05)  # 5% of TX_POWER per other user

        sinr = received_power_mw / (noise_power_mw + interference_mw)
        sinr = max(min(sinr, 1e6), 1e-3)  # SINR must be > 0 for log2

        # Effective bandwidth might be reduced by concurrent users
        effective_bandwidth = BANDWIDTH_RANGE[0] / concurrent_users

        # Data rate using Shannon capacity formula (in bits/sec)
        datarate = effective_bandwidth * np.log2(1 + sinr)
        datarate = max(datarate, 1000)  # Ensure minimum data rate

        transmission_time = self.task.size_bits / datarate

        # Processing time on the server
        # Consider available frequency based on server's current utilization
        available_freq = server.frequency * (1.0 - server_load)
        if available_freq <= 0:  # Server is fully utilized
            return float('inf')
        processing_time = self.task.required_cycles / available_freq

        # Total time for offloading (transmission + processing + result return)
        total_offload_time = transmission_time + processing_time + (transmission_time * OP_IP_RATIO)

        # If offloading exceeds deadline, assign very high potential
        if total_offload_time > self.task.deadline:
            return float('inf')

        # Energy consumption for transmission
        transmission_energy = BETA * self.task.size_bits

        # Server cost (can be adjusted based on preference for RSU/BS/Cloud)
        server_cost = {'RSU': 1.0, 'BS': 2.0, 'Cloud': 3.0}.get(getattr(server, 'server_type', 'RSU'), 2.0)

        # Combine time, energy, and server cost for the potential
        return (total_offload_time * WEIGHTS[0]) + (transmission_energy * WEIGHTS[1]) + server_cost

    def _calculate_v2v_offload_potential(self, target_vehicle: Vehicle) -> float:
        """Calculate the potential (cost) for offloading to another vehicle (V2V)."""
        distance = haversine(self.vehicle.position[0], self.vehicle.position[1],
                             target_vehicle.position[0], target_vehicle.position[1])

        # Ensure minimum distance
        distance = max(distance, 0.001)

        # Convert V2V_TX_POWER from Watts to dBm
        v2v_tx_power_dbm = 10 * np.log10(V2V_TX_POWER * 1000)

        log10_distance = np.log10(distance)
        path_loss_db = 38.02 + 20 * log10_distance  # Assuming same path loss model for V2V

        received_power_dbm = v2v_tx_power_dbm - path_loss_db

        noise_power_dbm = 10 * np.log10(NOISE_POWER * 1000)

        received_power_mw = 10 ** (received_power_dbm / 10)
        noise_power_mw = 10 ** (noise_power_dbm / 10)

        # SINR calculation for V2V
        sinr = received_power_mw / noise_power_mw  # Assuming no explicit V2V interference model for simplicity
        sinr = max(min(sinr, 1e6), 1e-3)

        datarate = V2V_BANDWIDTH * np.log2(1 + sinr)
        datarate = max(datarate, 1000)

        transmission_time = self.task.size_bits / datarate

        # Processing time on the target vehicle
        # Check target vehicle's availability based on its current utilization
        available_freq_v2v = target_vehicle.frequency * (1.0 - target_vehicle.utilization)
        if available_freq_v2v <= 0:  # Target vehicle is fully utilized
            return float('inf')
        processing_time = self.task.required_cycles / available_freq_v2v

        total_v2v_time = transmission_time + processing_time

        # If V2V offloading exceeds deadline, assign very high potential
        if total_v2v_time > self.task.deadline:
            return float('inf')

        # Energy consumption for V2V transmission
        transmission_energy = BETA * self.task.size_bits

        # V2V specific cost (can be lower than RSU/BS/Cloud as it's peer-to-peer)
        v2v_cost = 0.5  # Example cost, can be tuned

        return (total_v2v_time * WEIGHTS[0]) + (transmission_energy * WEIGHTS[1]) + v2v_cost

    def _find_eligible_v2v_candidates(self) -> list[Vehicle]:
        """Find vehicles within V2V range that can process the task before deadline."""
        # eligible_vehicles = []
        # for v in self.all_vehicles:
        #     distance = haversine(self.vehicle.position[0], self.vehicle.position[1],
        #                          v.position[0], v.position[1])
        #
        #     if distance <= V2V_RANGE_KM:
        #         # Check if the target vehicle is available and can meet the deadline
        #         # This check is simplified here; a full SimPy resource model for vehicles would be better
        #         # For now, we use the is_available method from the Vehicle class
        #         if v.is_available(self.task):  # This checks utilization and deadline
        #             eligible_vehicles.append(v)
        # return eligible_vehicles
        return [v for v in self.all_vehicles
                if v.id != self.vehicle.id
                and haversine(self.vehicle.position[0], self.vehicle.position[1], v.position[0], v.position[1]) <= V2V_RANGE_KM
                and v.utilization < 0.7  # Add utilization buffer
                and (v.frequency * (1 - v.utilization)) > (self.task.required_cycles / self.task.deadline)]

    def solve(self):
        """
        Solves the offloading problem using a game-theoretic approach, considering
        local, V2V, RSU, BS, and Cloud options.
        """
        options = []  # List of (potential_value, target_object) tuples
        targets = []  # List of actual target objects (None for local)

        # 1. Local Processing Option
        local_potential = self._calculate_local_potential()
        options.append(local_potential)
        targets.append(None)  # None represents local processing

        # 2. V2V Offloading Options
        eligible_v2v_candidates = self._find_eligible_v2v_candidates()
        for v_candidate in eligible_v2v_candidates:
            v2v_potential = self._calculate_v2v_offload_potential(v_candidate)
            options.append(v2v_potential)
            targets.append(v_candidate)

        # 3. RSU, BS, Cloud Offloading Options
        for server in self.available_servers:
            # For RSUs, add a mobility stability check here as per your original logic
            if server.server_type == 'RSU':
                distance = haversine(self.vehicle.position[0], self.vehicle.position[1],
                                     server.position[0], server.position[1])
                # Check if vehicle will stay in RSU range for the task deadline
                # Convert RSU_RANGE_KM to meters, distance to meters, speed to m/s
                # This is a simplified linear projection; a more complex one would consider direction.
                if self.vehicle.speed > 0:
                    time_to_leave_rsu = (RSU_RANGE_KM * 1000 - (distance * 1000)) / (self.vehicle.speed / 3.6)
                    if time_to_leave_rsu < self.task.deadline:
                        continue  # Skip this RSU if vehicle won't stay in range
                # If speed is 0 or it stays in range, proceed

            # Check if the server can accept the task (e.g., if it has capacity)
            # The server.can_accept_task() is a simple True for now in model.py
            # The potential function already considers server utilization.
            # So, we just check if the potential is not infinity.
            server_potential = self._calculate_offload_potential(server)
            options.append(server_potential)
            targets.append(server)

        # Convert potentials to numpy array
        potentials = np.array(options)

        # Handle cases where all options are infinitely costly (e.g., cannot meet deadline)
        if np.all(np.isinf(potentials)):
            # return None  # No feasible option, default to local (or fail)
            valid_options = [i for i, p in enumerate(potentials) if p < float('inf')]
            if valid_options:
                return targets[np.nanargmin([potentials[i] for i in valid_options])]

        # Apply softmax to convert potentials (costs) into probabilities
        # Lower potential (cost) should result in higher probability
        # We use -potentials because softmax works with higher values meaning higher probability
        # If any potential is inf, -inf will be used, resulting in 0 probability for that option.
        valid_potentials = -potentials / self.temperature

        # Replace -inf with a very small number to avoid issues with softmax on pure -inf arrays
        # np.exp(-inf) is 0, so this is generally fine, but explicit handling can prevent warnings
        valid_potentials[np.isinf(valid_potentials)] = -1e9  # A very small number

        probabilities = softmax(valid_potentials)

        # Select an option based on probabilities
        # np.random.choice requires probabilities to sum to 1.
        # If all options were inf, probabilities will be all 0, leading to an error.
        # This is handled by the np.all(np.isinf(potentials)) check above.

        # Ensure probabilities sum to 1, handling potential floating point inaccuracies
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum == 0:  # Should be caught by np.all(np.isinf(potentials))
            return None
        probabilities = probabilities / probabilities_sum

        chosen_idx = np.random.choice(len(targets), p=probabilities)
        return targets[chosen_idx]
