"""
DDPG Agent for joint service caching and computation offloading.

The agent operates on the LEAF infrastructure and makes decisions at each
time slot updating:
  - caching decisions for edge servers: c_{e,k} ∈ {0,1}  (binarised from continuous)
  - offloading ratios:  o_v ∈ [0,1],  o_e ∈ [0,1]

State vector (per time slot):
  - For each edge e: cache status of NK services (NK bits)
  - For each vehicle v: service requested (one-hot, NK bits) + position (normalised)
  - Effective SINR (normalised, 1 float)

Action vector (per time slot):
  - Edge caching: NUM_EDGES × NK continuous values ∈ [0,1] (binarised at threshold 0.5)
  - Offload ratios: NUM_VEHICLES × 2 values [o_v, o_e] ∈ [0,1]

Total state dim = NUM_EDGES×NK + NUM_VEHICLES×(NK+1) + 1
Total action dim = NUM_EDGES×NK + NUM_VEHICLES×2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from examples.vec_caching.settings import (
    NUM_EDGES, NUM_SERVICES, NUM_VEHICLES,
    ACTOR_LR, CRITIC_LR, TAU, GAMMA, BATCH_SIZE, HIDDEN_SIZE,
    NOISE_STD_INIT, NOISE_DECAY,
)
from examples.vec_caching.replay_buffer import ReplayBuffer


from examples.vec_caching.settings import USE_SUMO

# ─────────────────────────────────────────────────────────────────────────────
# State / Action dimensions
# ─────────────────────────────────────────────────────────────────────────────
"""State vector (per time slot) — Equation 36 of the paper:
  s_e(t) = { I_{v,k} (request indicator),
             c^V_{v,k} (vehicle caching indicators),
             B_{v,e} (bandwidth per vehicle),
             γ_{v,e} (SINR),
             c^E_{e,k} (edge caching indicators) }

Action vector (per time slot) — Equation 38:
  a_e(t) = { c^E_{e,k}, o^V_{v,k}, o^E_{e,k} }

State dim  = ρmax×NK (I) + ρmax×NK (c^V) + ρmax (B) + ρmax (γ) + NE×NK (c^E)
           = NUM_VEHICLES*(2*NUM_SERVICES + 2) + NUM_EDGES*NUM_SERVICES
Action dim = NUM_EDGES*NK + NUM_VEHICLES*2
"""

# State: per vehicle: NK (I) + NK (c^V) + 1 (B) + 1 (γ) + per edge: NK (c^E)
STATE_DIM = NUM_VEHICLES * (2 * NUM_SERVICES + 2) + NUM_EDGES * NUM_SERVICES
ACTION_DIM = NUM_EDGES * NUM_SERVICES + NUM_VEHICLES * 2


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Maps state → action ∈ [0,1]^{ACTION_DIM}.
    Architecture matches paper Fig 4: 512 → 256 → 128 → 64.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid(),   # output ∈ [0,1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Maps (state, action) → Q-value. Mirrors Actor depth."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# DDPG Agent
# ─────────────────────────────────────────────────────────────────────────────

class DDPGAgent:
    def __init__(self,
                 state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM,
                 replay_capacity: int = 10000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay = ReplayBuffer(replay_capacity)
        self.noise_std = NOISE_STD_INIT
        self.action_dim = action_dim

    # ── Public API ────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action (with exploration noise during training)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().squeeze(0)
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, 0.0, 1.0)
        return action

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self):
        """One DDPG update step. Returns (critic_loss, actor_loss) or None."""
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            a_next = self.actor_target(ns)
            q_next = self.critic_target(ns, a_next)
            q_target = r + GAMMA * (1 - d) * q_next
        q_pred = self.critic(s, a)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft target updates
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def decay_noise(self):
        self.noise_std = max(self.noise_std * NOISE_DECAY, 0.01)

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# State / Action helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_state(edge_servers, vehicles, mobility_manager, road_length, sinr_linear) -> np.ndarray:
    """Build normalised state vector per paper Equation 36.

    s_e = { I_{v,k}, c^V_{v,k}, B_{v,e}, γ_{v,e}, c^E_{e,k} }
    """
    from examples.vec_caching.settings import BANDWIDTH, MAX_SERVICES_VEHICLE
    state = []

    # 1) Per vehicle: I_{v,k} (request one-hot) + c^V_{v,k} (cache indicators)
    #    + B_{v,e} (bandwidth, normalised) + γ_{v,e} (SINR, normalised)
    n_tasks = sum(len(e.vehicles_in_range) for e in edge_servers)
    n_tasks = max(n_tasks, 1)
    bw_per_vehicle = BANDWIDTH / n_tasks
    bw_norm = min(bw_per_vehicle / BANDWIDTH, 1.0)  # normalise to [0,1]
    sinr_norm = min((sinr_linear - 1.0) / 5.0, 1.0) # normalise to [0,1]

    for v in vehicles:
        # I_{v,k}: request indicator (one-hot over NK services)
        service = v.current_task_service if v.current_task_service is not None else 0
        one_hot = [0.0] * NUM_SERVICES
        one_hot[service] = 1.0
        state.extend(one_hot)

        # c^V_{v,k}: vehicle caching indicators (NK bits)
        for k in range(NUM_SERVICES):
            state.append(1.0 if k in v.cached_services else 0.0)

        # B_{v,e}: normalised bandwidth allocated to this vehicle
        state.append(bw_norm)

        # γ_{v,e}: normalised SINR
        state.append(sinr_norm)

    # 2) Per edge: c^E_{e,k} (edge caching indicators)
    for edge in edge_servers:
        for k in range(NUM_SERVICES):
            state.append(1.0 if k in edge.cached_services else 0.0)

    return np.array(state, dtype=np.float32)



def parse_action(action: np.ndarray, edge_servers, vehicles):
    """
    Split raw DDPG action vector into caching decisions and offload ratios.

    Returns:
        edge_cache_decisions: dict {edge: [service_indices to cache]}
        offload_ratios: dict {vehicle: (o_v, o_e)}
    """
    idx = 0

    # Edge caching decisions (binarise at 0.5)
    edge_cache = {}
    for edge in edge_servers:
        from examples.vec_caching.settings import MAX_SERVICES_EDGE
        scores = action[idx: idx + NUM_SERVICES]
        idx += NUM_SERVICES
        # Select top-k services by score, up to storage limit
        ranked = sorted(range(NUM_SERVICES), key=lambda k: scores[k], reverse=True)
        edge_cache[edge] = ranked[:MAX_SERVICES_EDGE]

    # Offload ratios per vehicle
    offload = {}
    for v in vehicles:
        o_v = float(np.clip(action[idx], 0.0, 1.0))
        o_e = float(np.clip(action[idx + 1], 0.0, 1.0))
        idx += 2
        offload[v] = (o_v, o_e)

    return edge_cache, offload
