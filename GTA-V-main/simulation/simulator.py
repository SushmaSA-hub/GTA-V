import simpy
import random
import numpy as np
from core.parameters import *
from core.position import haversine, rsu_positions, base_station_positions, cloud_position, vehicle_positions
from core.model import Vehicle, Server, Task
from strategies.deco_offloader import DECOOffloader
from strategies.game_theory import PotentialGameOffloader
from strategies.greedy_offloader import GreedyOffloader
from strategies.local_baseline import LocalOnlyOffloader
from strategies.random_baseline import RandomOffloader


class AutonomousVehicleSimulator:
    """
    Main simulation engine for autonomous vehicle task offloading.
    Uses SimPy for discrete event simulation and game theory for offloading decisions.
    """

    def __init__(self, strategy='game_theory'):
        self.strategy = strategy
        self.env = simpy.Environment()
        self.servers = self.create_servers()
        self.vehicles = self.create_vehicles()
        self.metrics = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'local_processing': 0,
            'v2v_processing': 0,
            'offloaded_processing': 0,
            'offloaded_by_type': {'RSU': 0, 'BS': 0, 'Cloud': 0, 'V2V': 0},
            'task_latencies': [],
            'energy_consumption': [], # This list will now store energy for ALL tasks (completed or failed)
            'potential_local_latencies': [] # Local processing time for ALL generated tasks
        }

    def create_servers(self):
        """Create RSU, base station and cloud servers"""
        servers = []
        for i, pos in enumerate(rsu_positions):
            servers.append(Server(self.env, i, 'RSU', FREQ_RSU, pos))
        for i, pos in enumerate(base_station_positions):
            servers.append(Server(self.env, i, 'BS', FREQ_BS, pos))
        self.cloud = Server(self.env, 0, 'Cloud', FREQ_CLOUD, cloud_position)
        return servers

    def create_vehicles(self):
        """Create vehicles with initial positions"""
        return [Vehicle(self.env, i, pos) for i, pos in enumerate(vehicle_positions)]

    def get_servers_in_range(self, vehicle_position, vehicle, task=None):
        """
        Get servers that are in range of the vehicle and will remain in range for task duration.
        """
        available_servers = []
        vehicle_region = None
        for region, center in SUBREGION_CENTERS.items():
            if self.is_in_region(vehicle_position, center, SUBREGION_SIDE_DEG):
                vehicle_region = region
                break

        for server in self.servers:
            if server.server_type == 'RSU':
                distance = haversine(*vehicle_position, *server.position)
                if distance <= RSU_RANGE_KM:
                    available_servers.append(server)
            elif server.server_type == 'BS':
                if vehicle_region and self.is_in_region(server.position, SUBREGION_CENTERS[vehicle_region], SUBREGION_SIDE_DEG):
                    available_servers.append(server)
        self.cloud.position = cloud_position # Ensure cloud position is always correct
        available_servers.append(self.cloud)
        return available_servers

    def is_in_region(self, position, center, region_side):
        """Check if a position is within a region"""
        lat, lon = position
        center_lat, center_lon = center
        half_side = region_side / 2
        return (center_lat - half_side <= lat <= center_lat + half_side) and \
            (center_lon - half_side <= lon <= center_lon + half_side)

    def vehicle_process(self, vehicle_id):
        """SimPy process for vehicle behavior"""
        vehicle = self.vehicles[vehicle_id]

        while True:
            task = vehicle.generate_task()
            print(
                f"{self.env.now:.2f}: Vehicle {vehicle_id} generated task of size {task.size_bits / 8e6:.2f}MB with deadline {task.deadline:.2f}s and vehicle power is {vehicle.frequency/1e9:.2f} GHz")

            # Record potential local processing time for this task
            # This is the baseline for total tasks generated
            self.metrics['potential_local_latencies'].append(task.required_cycles / vehicle.frequency)

            vehicle.move()

            potential_servers = self.get_servers_in_range(vehicle.position, vehicle, task)
            all_vehicles_for_v2v = [v for v in self.vehicles if v.id != vehicle.id]

            # Strategy-based offloader selection
            offloader = None # Initialize offloader
            if self.strategy == PotentialGameOffloader:
                offloader = PotentialGameOffloader(self.env, vehicle, task, potential_servers, all_vehicles_for_v2v)
            elif self.strategy == DECOOffloader:
                offloader = DECOOffloader(self.env, vehicle, task, potential_servers)
            elif self.strategy == LocalOnlyOffloader:
                offloader = LocalOnlyOffloader(self.env, vehicle, task, potential_servers)
            elif self.strategy == RandomOffloader:
                offloader = RandomOffloader(self.env, vehicle, task, potential_servers)
            elif self.strategy == GreedyOffloader:
                offloader = GreedyOffloader(self.env, vehicle, task, potential_servers)
            else:
                # Default to Proposed if strategy string is used, or if it's an unknown strategy
                offloader = PotentialGameOffloader(self.env, vehicle, task, potential_servers, all_vehicles_for_v2v)

            selected_target = offloader.solve()

            # --- Processing Decision & Metric Recording ---
            # Initialize variables to ensure they are always defined for final metric recording
            total_time = float('inf') # Default to inf, will be updated if completed
            energy_spent = 0.0
            task_outcome_message = "UNACCOUNTED OUTCOME" # Default message, should always be overwritten

            if selected_target is None:
                # Decision is to process locally (either by explicit choice or no offload option found)
                processing_time = task.required_cycles / vehicle.frequency
                energy_spent = ALPHA * task.required_cycles

                print(f"{self.env.now:.2f}: Vehicle {vehicle_id} processing locally, estimated time: {processing_time:.2f}s")
                yield self.env.timeout(processing_time)
                total_time = processing_time

                if total_time <= task.deadline:
                    self.metrics['local_processing'] += 1
                    self.metrics['completed_tasks'] += 1
                    task_outcome_message = f"completed task locally (actual time: {total_time:.2f}s)"
                else:
                    self.metrics['failed_tasks'] += 1
                    task_outcome_message = f"failed task locally (exceeded deadline: {total_time:.2f}s vs {task.deadline:.2f}s)"

            elif isinstance(selected_target, Vehicle):
                # V2V offloading
                target_vehicle = selected_target
                distance = haversine(*vehicle.position, *target_vehicle.position)
                transmission_time = self.calculate_v2v_transmission_time(task, distance)
                energy_spent = V2V_TX_POWER * transmission_time

                print(f"{self.env.now:.2f}: Offloading task from {vehicle_id} to Vehicle {target_vehicle.id}")

                yield self.env.timeout(transmission_time)
                processing_time = task.required_cycles / target_vehicle.frequency
                yield self.env.timeout(processing_time)

                total_time = transmission_time + processing_time
                if total_time <= task.deadline:
                    self.metrics['completed_tasks'] += 1
                    self.metrics['v2v_processing'] += 1
                    self.metrics['offloaded_by_type']['V2V'] += 1
                    task_outcome_message = f"task completed on Vehicle {target_vehicle.id} (actual time: {total_time:.2f}s)"
                else:
                    self.metrics['failed_tasks'] += 1
                    task_outcome_message = f"task failed on Vehicle {target_vehicle.id} (exceeded deadline: {total_time:.2f}s vs {task.deadline:.2f}s)"

            else: # selected_target is a Server (RSU, BS, or Cloud)
                selected_server = selected_target
                try:
                    with selected_server.request() as req:
                        yield req  # Request server resource

                        distance = haversine(*vehicle.position, *selected_server.position)
                        transmission_time = self.calculate_transmission_time(task, distance)
                        energy_spent = TX_POWER * transmission_time

                        print(f"{self.env.now:.2f}: Offloading task from {vehicle_id} to {selected_server.server_type} {selected_server.server_id}")

                        yield self.env.timeout(transmission_time) # Simulate uplink transmission
                        processing_time = selected_server.calculate_processing_time(task)
                        yield self.env.timeout(processing_time) # Simulate processing on server
                        yield self.env.timeout(transmission_time * OP_IP_RATIO) # Simulate downlink transmission for results

                        total_time = transmission_time + processing_time + (transmission_time * OP_IP_RATIO)

                        if total_time <= task.deadline:
                            self.metrics['offloaded_processing'] += 1
                            self.metrics['offloaded_by_type'][selected_server.server_type] += 1
                            self.metrics['completed_tasks'] += 1
                            task_outcome_message = f"task completed on {selected_server.server_type} (actual time: {total_time:.2f}s)"
                        else:
                            self.metrics['failed_tasks'] += 1
                            task_outcome_message = f"task failed on {selected_server.server_type} (exceeded deadline: {total_time:.2f}s vs {task.deadline:.2f}s)"

                except simpy.Interrupt:
                    self.metrics['failed_tasks'] += 1
                    # If interrupted, record energy for transmission up to that point (simplified)
                    energy_spent = TX_POWER * self.calculate_transmission_time(task, distance)
                    total_time = float('inf') # Mark as infinite time for failed/interrupted tasks
                    task_outcome_message = f"task interrupted by {selected_server.server_type}"

            # --- Final Metric Recording for EACH generated task ---
            # These lines are crucial and MUST be outside the if/elif/else block
            # to ensure every generated task has its latency and energy recorded.
            self.metrics['task_latencies'].append(total_time)
            self.metrics['energy_consumption'].append(energy_spent)
            print(f"{self.env.now:.2f}: Vehicle {vehicle_id} {task_outcome_message}")

            yield self.env.timeout(random.expovariate(1 / 5)) # Wait before next task

    def calculate_transmission_time(self, task, distance):
        """
        Calculate transmission time for V2I (Vehicle-to-Infrastructure) communication.
        Corrected to handle dBm for TX_POWER and path loss.
        """
        distance = max(distance, 0.001)
        tx_power_dbm = 10 * np.log10(TX_POWER * 1000)
        log10_distance = np.log10(distance)
        path_loss_db = 38.02 + 20 * log10_distance
        received_power_dbm = tx_power_dbm - path_loss_db
        noise_power_dbm = 10 * np.log10(NOISE_POWER * 1000)
        received_power_mw = 10**(received_power_dbm / 10)
        noise_power_mw = 10**(noise_power_dbm / 10)
        sinr = received_power_mw / noise_power_mw
        sinr = max(min(sinr, 1e6), 1e-3)
        data_rate = BANDWIDTH_RANGE[0] * np.log2(1 + sinr)
        data_rate = max(data_rate, 1000)
        return max(task.size_bits / data_rate, 0.001)

    def run(self, until=SIM_TIME):
        """Run the simulation for a specified time"""
        for i in range(len(self.vehicles)):
            self.env.process(self.vehicle_process(i))
        self.env.run(until=until)
        return self.metrics

    def calculate_v2v_transmission_time(self, task, distance):
        """
        Calculate V2V transmission time using similar logic to server comms.
        Corrected to handle dBm for V2V_TX_POWER and path loss.
        """
        distance = max(distance, 0.001)
        v2v_tx_power_dbm = 10 * np.log10(V2V_TX_POWER * 1000)
        log10_distance = np.log10(distance)
        path_loss_db = 38.02 + 20 * log10_distance
        received_power_dbm = v2v_tx_power_dbm - path_loss_db
        noise_power_dbm = 10 * np.log10(NOISE_POWER * 1000)
        received_power_mw = 10**(received_power_dbm / 10)
        noise_power_mw = 10**(noise_power_dbm / 10)
        sinr = received_power_mw / noise_power_mw
        sinr = max(min(sinr, 1e6), 1e-3)
        datarate = V2V_BANDWIDTH * np.log2(1 + sinr)
        datarate = max(datarate, 1000)
        return max(task.size_bits / datarate, 0.001)
