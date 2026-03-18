from dataclasses import dataclass, field
import simpy
import random
from .parameters import *
from .position import rsu_positions, base_station_positions, cloud_position, vehicle_positions


# --- Task entity ---
@dataclass
class Task:
    vehicle_id: int
    size_bits: float
    deadline: float
    created_time: float
    required_cycles: float = 0.0
    assigned_to: str = "Unassigned"
    offload_position: tuple = (0.0, 0.0)
    status: str = "Pending"

    def __post_init__(self):
        self.required_cycles = self.size_bits * RESOURCE_PER_BIT


# --- Server entity ---
class Server(simpy.Resource):
    def __init__(self, env, server_id, server_type, frequency, position):
        super().__init__(env, capacity=1)
        self.server_id = server_id
        self.server_type = server_type
        self.frequency = frequency    # In Hz
        self.position = position      # (lat, lon)
        self.utilization = 0.0
        self.queue = []

    def calculate_processing_time(self, task):
        # Task cycles / server freq (Hz)
        # return task.required_cycles / self.frequency
        # Consider current utilization
        available_freq = self.frequency * (1 - self.utilization)
        return task.required_cycles / available_freq if available_freq > 0 else float('inf')

    def request_freq(self, required_freq):
        if self.utilization + required_freq > self.frequency:
            raise simpy.Interrupt("Insufficient capacity")
        self.utilization += required_freq
        return self.request()

    def can_accept_task(self, task, deadline=None):
        # Simple check: always True for now; can implement utilization logic
        return True


# --- Vehicle entity ---
class Vehicle:
    def __init__(self, env, vehicle_id, position, speed=None):
        self.env = env
        self.id = vehicle_id
        self.position = position
        self.speed = speed if speed else random.uniform(*VEHICLE_SPEED_RANGE)  # km/h
        self.current_task = None
        self.frequency = random.uniform(*FREQ_LOCAL_RANGE)
        # Add this attribute to store offload position
        self.offload_position = position
        # Make sure deadline is accessible from the Vehicle class
        self.deadline = None  # Will be set when generating tasks
        self.utilization = 0.0  # 0.0 = free, 1.0 = fully busy
        self.task_queue = []  # Tasks assigned by other vehicles

    def is_available(self, task):
        """Check if this vehicle can accept an external task within deadline"""
        available_freq = self.frequency * (1 - self.utilization)
        processing_time = task.required_cycles / available_freq
        return (processing_time <= task.deadline) and (self.utilization < 1.0)

    def generate_task(self):
        size_bits = random.uniform(*TASK_SIZE_RANGE)
        deadline = random.uniform(*TASK_DEADLINE_RANGE)
        return Task(
            vehicle_id=self.id,
            size_bits=size_bits,
            deadline=deadline,
            created_time=self.env.now
        )

    def calculate_local_time(self, task):
        """Calculate time to process task locally"""
        return task.required_cycles / self.frequency

    def move(self):
        """Update vehicle position based on speed"""
        delta = (self.speed / 111000) * (1 / 3600)  # km/s to degrees/second
        self.position = (
            self.position[0] + random.uniform(-delta, delta),
            self.position[1] + random.uniform(-delta, delta)
        )
