"""
MA-DDPG with Explicit Communication Protocol (MA-DDPG-Comm).

Extends the basic CTDE MA-DDPG with three explicit coordination mechanisms:

  1. COMMUNICATION PROTOCOL — MessageEncoder
       Each agent e encodes its local state into a compact intent vector:
           m_e(t) = Encoder(s_e(t))  ∈ R^{MSG_DIM}
       These messages are broadcast to all other agents over a virtual
       communication channel (models RSU-to-RSU signaling).

  2. SHARED INTENT SIGNALING — Message Aggregation
       Each agent aggregates received messages via mean-pooling:
           m̄_e(t) = mean({m_{e'}(t)}, e'≠e)
       The actor conditions on BOTH local state AND neighbour intent:
           a_e(t) = Actor([s_e(t) ‖ m̄_e(t)])
       Agents therefore know their neighbours' load and cache intentions
       BEFORE making their own decision — this is explicit intent signaling.

  3. CONFLICT-RESOLUTION — Coordination Penalty
       After all agents commit to cache actions, a conflict penalty is added:
           r_coord(t) = -λ_c · Σ_k max(0, Σ_e 𝟙[k ∈ C_e] − 1)
       This penalises any service cached by more than one edge (herding),
       pushing agents to learn complementary, non-overlapping cache sets.
       A priority tiebreaker (highest local demand wins) resolves ties.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  s_e(t) ──► MessageEncoder ──► m_e ─► broadcast ──► m̄_{e'} │
  │  s_e(t) ──────────────────────────────────────────────► │
  │                                       [s_e ‖ m̄_e] ──► CommActor ──► a_e │
  │  Centralised Critic Q(s_global, a_global)               │
  │  r = -avg_delay − λ_c · cache_conflict_count            │
  └─────────────────────────────────────────────────────────┘

References:
  - CommNet: Sukhbaatar et al., "Learning Multiagent Communication with
    Backpropagation", NeurIPS 2016.
  - DIAL: Foerster et al., "Learning to Communicate with Deep Multi-Agent
    Reinforcement Learning", NeurIPS 2016.
  - MADDPG: Lowe et al., "Multi-Agent Actor-Critic for Mixed
    Cooperative-Competitive Environments", NeurIPS 2017.
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
    ACTOR_LR, CRITIC_LR, TAU, GAMMA, BATCH_SIZE,
    NOISE_STD_INIT, NOISE_DECAY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────────────────

NV_PER_EDGE = max(1, NUM_VEHICLES // NUM_EDGES)

# Local state per agent: NV_LOCAL*(2*NK + 2) + NK
LOCAL_STATE_DIM  = NV_PER_EDGE * (2 * NUM_SERVICES + 2) + NUM_SERVICES
# Local action per agent: NK (cache) + NV_LOCAL*2 (offload)
LOCAL_ACTION_DIM = NUM_SERVICES + NV_PER_EDGE * 2

# Message vector dimension (communication channel width)
MSG_DIM = 16

# Augmented input to actor: local state + aggregated neighbour messages
AUGMENTED_STATE_DIM = LOCAL_STATE_DIM + MSG_DIM

# Global tensors (for centralised critic)
GLOBAL_STATE_DIM  = LOCAL_STATE_DIM  * NUM_EDGES
GLOBAL_ACTION_DIM = LOCAL_ACTION_DIM * NUM_EDGES

# Conflict penalty weight λ_c
LAMBDA_COORD = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class MessageEncoder(nn.Module):
    """
    Communication Protocol: encodes local state → intent message.

    Each edge server e computes m_e = Encoder(s_e) before acting.
    The message is a compact summary of:
      - which services the agent intends to cache (cache plan)
      - how loaded the agent currently is (load signal)
    This is broadcast to all neighbours before their actors run.
    """

    def __init__(self, local_state_dim: int = LOCAL_STATE_DIM,
                 msg_dim: int = MSG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, msg_dim),
            nn.Tanh(),          # bounded message values ∈ (-1, 1)
        )

    def forward(self, local_state: torch.Tensor) -> torch.Tensor:
        return self.net(local_state)


class CommActor(nn.Module):
    """
    Shared-Intent Actor: decides action from [local_state ‖ aggregated_msgs].

    By conditioning on neighbour messages, the actor knows what other edges
    intend to cache and how loaded they are — enabling coordinated decisions
    without run-time V2I/V2V communication overhead at inference.
    """

    def __init__(self, augmented_state_dim: int = AUGMENTED_STATE_DIM,
                 local_action_dim: int = LOCAL_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(augmented_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, local_action_dim),
            nn.Sigmoid(),   # output ∈ [0, 1]
        )

    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        return self.net(augmented_state)


class CentralisedCritic(nn.Module):
    """Shared critic: (global_state, global_action) → Q-value."""

    def __init__(self, global_state_dim: int = GLOBAL_STATE_DIM,
                 global_action_dim: int = GLOBAL_ACTION_DIM):
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hard_copy(src: nn.Module, dst: nn.Module):
    dst.load_state_dict(src.state_dict())

def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
    for sp, dp in zip(src.parameters(), dst.parameters()):
        dp.data.copy_(tau * sp.data + (1 - tau) * dp.data)


# ─────────────────────────────────────────────────────────────────────────────
# MA-DDPG-Comm System
# ─────────────────────────────────────────────────────────────────────────────

class MADDPGCommSystem:
    """
    MA-DDPG with Explicit Communication Protocol.

    Per decision step:
      1. Each agent encodes a message from its local state (MessageEncoder).
      2. Messages are broadcast and mean-pooled across all agents.
      3. Each agent acts on [local_state ‖ aggregated_msg] (CommActor).
      4. A conflict penalty penalises duplicate caching decisions.
      5. Centralised Critic trains all actors cooperatively via CTDE.
    """

    def __init__(self,
                 n_agents: int = NUM_EDGES,
                 local_state_dim: int = LOCAL_STATE_DIM,
                 local_action_dim: int = LOCAL_ACTION_DIM,
                 msg_dim: int = MSG_DIM,
                 replay_capacity: int = 10_000):

        self.n_agents = n_agents
        self.msg_dim  = msg_dim
        self.device   = torch.device("cpu")

        # ── Message Encoders (one per agent) ─────────────────────────────────
        # These implement the COMMUNICATION PROTOCOL
        self.encoders      = []
        self.encoders_tgt  = []
        self.encoder_opts  = []
        for _ in range(n_agents):
            enc  = MessageEncoder(local_state_dim, msg_dim).to(self.device)
            enct = MessageEncoder(local_state_dim, msg_dim).to(self.device)
            _hard_copy(enc, enct)
            self.encoders.append(enc)
            self.encoders_tgt.append(enct)
            self.encoder_opts.append(
                optim.Adam(enc.parameters(), lr=ACTOR_LR)
            )

        # ── Communication Actors (one per agent) ─────────────────────────────
        aug_dim = local_state_dim + msg_dim
        self.actors      = []
        self.actors_tgt  = []
        self.actor_opts  = []
        for _ in range(n_agents):
            a  = CommActor(aug_dim, local_action_dim).to(self.device)
            at = CommActor(aug_dim, local_action_dim).to(self.device)
            _hard_copy(a, at)
            self.actors.append(a)
            self.actors_tgt.append(at)
            self.actor_opts.append(optim.Adam(a.parameters(), lr=ACTOR_LR))

        # ── Shared Centralised Critic ─────────────────────────────────────────
        gs = local_state_dim  * n_agents
        ga = local_action_dim * n_agents
        self.critic     = CentralisedCritic(gs, ga).to(self.device)
        self.critic_tgt = CentralisedCritic(gs, ga).to(self.device)
        _hard_copy(self.critic, self.critic_tgt)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # ── Replay Buffer ─────────────────────────────────────────────────────
        self.buffer: deque = deque(maxlen=replay_capacity)

        # ── Exploration noise ─────────────────────────────────────────────────
        self.noise_std = NOISE_STD_INIT

    # ── Step 1-3: Communication + Action ─────────────────────────────────────

    def _encode_messages(self, local_states: list,
                         use_target: bool = False) -> torch.Tensor:
        """
        Step 1: Encode intent messages from local states.
        Returns tensor of shape (n_agents, msg_dim).
        """
        encoders = self.encoders_tgt if use_target else self.encoders
        msgs = []
        for i, s in enumerate(local_states):
            st = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            msgs.append(encoders[i](st))          # (1, msg_dim)
        return torch.cat(msgs, dim=0)             # (n_agents, msg_dim)

    def _aggregate_messages(self, all_msgs: torch.Tensor) -> torch.Tensor:
        """
        Step 2: Mean-pool neighbour messages for each agent.
        Returns tensor of shape (n_agents, msg_dim).
        """
        n = self.n_agents
        aggregated = torch.zeros_like(all_msgs)
        for i in range(n):
            # exclude own message
            others = [all_msgs[j] for j in range(n) if j != i]
            aggregated[i] = torch.stack(others).mean(dim=0)
        return aggregated

    def get_actions(self, local_states: list) -> list:
        """
        Full communication + action pipeline (inference).

        Steps:
          1. Each agent encodes a message (COMMUNICATION PROTOCOL).
          2. Messages broadcast and aggregated (INTENT SIGNALING).
          3. Each actor conditions on [local_state ‖ aggregated_msg].
        """
        with torch.no_grad():
            # Step 1 & 2: communicate
            all_msgs = self._encode_messages(local_states)
            agg_msgs = self._aggregate_messages(all_msgs)   # (n_agents, msg_dim)

            # Step 3: act on augmented observation
            actions = []
            for i, s in enumerate(local_states):
                st  = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                aug = torch.cat([st, agg_msgs[i].unsqueeze(0)], dim=-1)
                a   = self.actors[i](aug).squeeze(0).cpu().numpy()
                a   = np.clip(a + np.random.normal(0, self.noise_std, a.shape), 0, 1)
                actions.append(a)
        return actions

    # ── Step 4: Conflict-Resolution ──────────────────────────────────────────

    @staticmethod
    def conflict_penalty(actions: list, edge_servers) -> float:
        """
        Step 4: Conflict-resolution penalty.

        Computes how many services are cached by more than one edge.
        Returns the penalty value r_coord = -λ_c * Σ_k max(0, count_k - 1).

        This incentivises agents to learn complementary, non-overlapping
        cache sets — eliminating herding behaviour.
        """
        from examples.vec_caching.settings import MAX_SERVICES_EDGE
        # Decode cache intention from each agent's action (top-k by score)
        service_counts = [0] * NUM_SERVICES
        for i, action in enumerate(actions):
            scores = action[:NUM_SERVICES]
            ranked = sorted(range(NUM_SERVICES),
                            key=lambda k: scores[k], reverse=True)
            for k in ranked[:MAX_SERVICES_EDGE]:
                service_counts[k] += 1

        redundancy = sum(max(0, c - 1) for c in service_counts)
        return -LAMBDA_COORD * redundancy

    # ── Replay ────────────────────────────────────────────────────────────────

    def store(self, local_states, actions, reward, next_local_states, done):
        self.buffer.append((
            np.concatenate(local_states),
            np.concatenate(actions),
            float(reward),
            np.concatenate(next_local_states),
            float(done),
        ))

    def decay_noise(self):
        self.noise_std = max(0.01, self.noise_std * NOISE_DECAY)

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self):
        """CTDE update: critic + actors + encoders."""
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
            # Build target augmented states via target encoders
            next_local_states_list = [
                gns[:, i * lsd:(i + 1) * lsd] for i in range(self.n_agents)
            ]
            # Encode target messages for each agent
            tgt_msgs = torch.stack([
                self.encoders_tgt[i](next_local_states_list[i])
                for i in range(self.n_agents)
            ], dim=1)                           # (B, n_agents, msg_dim)

            # Aggregate target messages
            tgt_agg = []
            for i in range(self.n_agents):
                others = [tgt_msgs[:, j, :] for j in range(self.n_agents) if j != i]
                tgt_agg.append(torch.stack(others, dim=1).mean(dim=1))  # (B, msg_dim)

            # Build target actions
            target_actions = []
            for i in range(self.n_agents):
                ns_i  = next_local_states_list[i]
                aug_i = torch.cat([ns_i, tgt_agg[i]], dim=-1)
                target_actions.append(self.actors_tgt[i](aug_i))
            gna = torch.cat(target_actions, dim=-1)
            y   = rw + GAMMA * (1 - dn) * self.critic_tgt(gns, gna)

        q = self.critic(gs, ga)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor + Encoder updates ───────────────────────────────────────────
        # Each agent gets its own freshly-computed graph to avoid autograd
        # in-place modification errors when sharing tensors across backward passes.
        curr_local = [gs[:, i * lsd:(i + 1) * lsd] for i in range(self.n_agents)]

        for i in range(self.n_agents):
            # Re-encode messages fresh for this agent's backward pass
            msgs_i = [self.encoders[j](curr_local[j]) for j in range(self.n_agents)]

            # Aggregate: mean of OTHER agents' messages (detached except agent i)
            agg_for_agent = []
            for k in range(self.n_agents):
                others = []
                for j in range(self.n_agents):
                    if j == k:
                        continue
                    # keep encoder[i]'s output live; detach everyone else
                    m = msgs_i[j] if j == i else msgs_i[j].detach()
                    others.append(m)
                agg_for_agent.append(torch.stack(others, dim=1).mean(dim=1))

            # Build joint action
            actions_i = []
            for j in range(self.n_agents):
                aug_j = torch.cat([curr_local[j], agg_for_agent[j].detach()
                                   if j != i else agg_for_agent[j]], dim=-1)
                if j == i:
                    actions_i.append(self.actors[i](aug_j))
                else:
                    actions_i.append(self.actors[j](aug_j).detach())
            ga_i = torch.cat(actions_i, dim=-1)

            actor_loss = -self.critic(gs, ga_i).mean()
            self.actor_opts[i].zero_grad()
            self.encoder_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()
            self.encoder_opts[i].step()

        # ── Soft target updates ───────────────────────────────────────────────
        for i in range(self.n_agents):
            _soft_update(self.actors[i],  self.actors_tgt[i],  TAU)
            _soft_update(self.encoders[i], self.encoders_tgt[i], TAU)
        _soft_update(self.critic, self.critic_tgt, TAU)
