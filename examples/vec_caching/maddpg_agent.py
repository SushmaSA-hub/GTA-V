"""
Multi-Agent DDPG (MADDPG) for VEC Joint Caching and Offloading.

Architecture (CTDE — Centralised Training, Decentralised Execution):
  - N_AGENTS = NUM_EDGES (one actor per edge server)
  - Each Actor observes LOCAL edge state (local vehicles + its own cache)
  - One SHARED Centralised Critic observes GLOBAL state + ALL agents' actions
  - Shared Replay Buffer stores (global_state, all_actions, reward, next_global_state)

Local state per agent i (per time slot):
  - I_{v,k}   request indicators for vehicles in range of edge i  (NV_LOCAL × NK)
  - c^V_{v,k} vehicle caching indicators                          (NV_LOCAL × NK)
  - B_{v,e}   bandwidth per local vehicle                         (NV_LOCAL)
  - γ_{v,e}   SINR per local vehicle                             (NV_LOCAL)
  - c^E_{e,k} edge i caching indicators                          (NK)

Local action per agent i:
  - c^E_{e,k}  caching decision for edge i                       (NK)
  - o^V_{v,k}  offloading ratio for local vehicles               (NV_LOCAL × 2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from examples.vec_caching.settings import (
    NUM_EDGES, NUM_SERVICES, NUM_VEHICLES,
    ACTOR_LR, CRITIC_LR, TAU, GAMMA, BATCH_SIZE, HIDDEN_SIZE,
    NOISE_STD_INIT, NOISE_DECAY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────────────────

# Vehicles split evenly across edges
NV_PER_EDGE = max(1, NUM_VEHICLES // NUM_EDGES)

# Local state per agent: NV_LOCAL*(2*NK + 2) + NK
LOCAL_STATE_DIM  = NV_PER_EDGE * (2 * NUM_SERVICES + 2) + NUM_SERVICES
# Local action per agent: NK (cache) + NV_LOCAL*2 (offload)
LOCAL_ACTION_DIM = NUM_SERVICES + NV_PER_EDGE * 2

# Global state = concatenated local states (all agents)
GLOBAL_STATE_DIM  = LOCAL_STATE_DIM  * NUM_EDGES
# Global action = concatenated local actions (all agents)
GLOBAL_ACTION_DIM = LOCAL_ACTION_DIM * NUM_EDGES


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class LocalActor(nn.Module):
    """Per-edge actor: local_state → local_action ∈ [0,1]."""

    def __init__(self, local_state_dim: int, local_action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, local_action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CentralisedCritic(nn.Module):
    """Shared critic: (global_state, global_action) → Q-value."""

    def __init__(self, global_state_dim: int, global_action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim + global_action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, global_state: torch.Tensor,
                global_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_state, global_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# MADDPG
# ─────────────────────────────────────────────────────────────────────────────

def _hard_copy(src: nn.Module, dst: nn.Module):
    dst.load_state_dict(src.state_dict())

def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
    for sp, dp in zip(src.parameters(), dst.parameters()):
        dp.data.copy_(tau * sp.data + (1 - tau) * dp.data)


class MADDPGSystem:
    """
    MADDPG with NUM_EDGES agents (one Actor per edge server).
    Single shared centralised Critic for cooperative training.
    """

    def __init__(self,
                 n_agents: int = NUM_EDGES,
                 local_state_dim: int = LOCAL_STATE_DIM,
                 local_action_dim: int = LOCAL_ACTION_DIM,
                 replay_capacity: int = 10_000):

        self.n_agents = n_agents
        self.device = torch.device("cpu")

        # ── Actors (one per agent) ────────────────────────────────────────────
        self.actors  = []
        self.actors_target = []
        self.actor_opts    = []
        for _ in range(n_agents):
            a  = LocalActor(local_state_dim, local_action_dim).to(self.device)
            at = LocalActor(local_state_dim, local_action_dim).to(self.device)
            _hard_copy(a, at)
            self.actors.append(a)
            self.actors_target.append(at)
            self.actor_opts.append(optim.Adam(a.parameters(), lr=ACTOR_LR))

        # ── Shared Centralised Critic ─────────────────────────────────────────
        gs = local_state_dim  * n_agents
        ga = local_action_dim * n_agents
        self.critic        = CentralisedCritic(gs, ga).to(self.device)
        self.critic_target = CentralisedCritic(gs, ga).to(self.device)
        _hard_copy(self.critic, self.critic_target)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # ── Replay Buffer ─────────────────────────────────────────────────────
        self.buffer: deque = deque(maxlen=replay_capacity)

        # ── Exploration noise ─────────────────────────────────────────────────
        self.noise_std = NOISE_STD_INIT

    # ── Action selection ──────────────────────────────────────────────────────

    def get_actions(self, local_states: list) -> list:
        """Given list of local states (one per agent), return list of noisy actions."""
        actions = []
        for i, s in enumerate(local_states):
            st = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a = self.actors[i](st).squeeze(0).cpu().numpy()
            a = np.clip(a + np.random.normal(0, self.noise_std, a.shape), 0, 1)
            actions.append(a)
        return actions

    def decay_noise(self):
        self.noise_std = max(0.01, self.noise_std * NOISE_DECAY)

    # ── Replay ────────────────────────────────────────────────────────────────

    def store(self, local_states, actions, reward, next_local_states, done):
        self.buffer.append((
            np.concatenate(local_states),    # global state
            np.concatenate(actions),          # global action
            reward,
            np.concatenate(next_local_states),
            float(done),
        ))

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.buffer, BATCH_SIZE)
        gs, ga, rw, gns, dn = zip(*batch)

        gs  = torch.FloatTensor(np.array(gs)).to(self.device)
        ga  = torch.FloatTensor(np.array(ga)).to(self.device)
        rw  = torch.FloatTensor(rw).unsqueeze(1).to(self.device)
        gns = torch.FloatTensor(np.array(gns)).to(self.device)
        dn  = torch.FloatTensor(dn).unsqueeze(1).to(self.device)

        lsd = LOCAL_STATE_DIM
        lad = LOCAL_ACTION_DIM

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            # Build target actions from target actors
            target_actions = []
            for i in range(self.n_agents):
                s_i = gns[:, i*lsd:(i+1)*lsd]
                target_actions.append(self.actors_target[i](s_i))
            gna = torch.cat(target_actions, dim=-1)
            y = rw + GAMMA * (1 - dn) * self.critic_target(gns, gna)

        q = self.critic(gs, ga)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor updates ─────────────────────────────────────────────────────
        for i in range(self.n_agents):
            # Recompute action only for agent i; keep others' actions detached
            actions_i = []
            for j in range(self.n_agents):
                s_j = gs[:, j*lsd:(j+1)*lsd]
                if j == i:
                    actions_i.append(self.actors[i](s_j))
                else:
                    actions_i.append(self.actors[j](s_j).detach())
            ga_i = torch.cat(actions_i, dim=-1)

            actor_loss = -self.critic(gs, ga_i).mean()
            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        # ── Soft target updates ───────────────────────────────────────────────
        for i in range(self.n_agents):
            _soft_update(self.actors[i], self.actors_target[i], TAU)
        _soft_update(self.critic, self.critic_target, TAU)
