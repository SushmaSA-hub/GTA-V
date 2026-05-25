"""
Experience Replay Buffer for DDPG training.

Implements Algorithm 1 from the paper:
- Buffer capacity = D
- When full: randomly replace an existing transition (NOT FIFO)
- Sampling: uniformly random mini-batch of N transitions
"""

import random
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Replay buffer with random replacement (matches Algorithm 1, lines 10-13)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []  # list for O(1) random access by index
        self._full = False

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        if not self._full:
            self.buffer.append(transition)
            if len(self.buffer) == self.capacity:
                self._full = True
        else:
            # Algorithm 1, line 13: randomly replace a transition when buffer is full
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = transition

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
