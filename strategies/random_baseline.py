import random

class RandomOffloader:
    """Baseline: Randomly offload to any available server or process locally."""
    def __init__(self, env, vehicle, task, available_servers):
        self.available_servers = available_servers
    def solve(self):
        if not self.available_servers:
            return None
        return random.choice(self.available_servers + [None])
