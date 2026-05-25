"""
Simulation parameters from Table III of the paper:
"Joint Service Caching and Computation Offloading Scheme Based on
 Deep Reinforcement Learning in Vehicular Edge Computing Systems"
IEEE Transactions on Vehicular Technology, 2023.
"""

import numpy as np

# ============================================================
# System Parameters (Table III)
# ============================================================
NUM_EDGES = 3            # NE: Number of edge servers
NUM_SERVICES = 5         # NK: Number of service types
RHO = 4                  # ρ: vehicle density per edge (default, in [2,5])
NUM_VEHICLES = RHO * NUM_EDGES  # Default: ρ=4 → 12 vehicles
NUM_EPISODES = 400       # Training episodes (paper uses ~370-400)

# Time
T_END = 40               # t_end: Total time slots per episode
DELTA_T = 30.0           # Δt: Duration of each time slot (s)

# ============================================================
# Communication Parameters (Table III)
# ============================================================
BANDWIDTH = 20e6                # B: Total bandwidth (Table III: 20 MHz)
EDGE_COMM_RANGE = 500.0         # Edge communication range: 500 m

# SINR fixed at 4~5 dB (Table III)
SINR_DB_MIN = 4.0
SINR_DB_MAX = 5.0
SINR_LINEAR_AVG = 10 ** (4.5 / 10)   # ≈ 2.818

R_EDGE = 15e6                   # R_edge: edge-to-edge rate 15 Mbps
R_CLOUD = 10e6                  # R_cloud: edge-to-cloud rate 10 Mbps

TX_POWER_EDGE = 1.0             # p_edge: edge-to-edge TX power (W)
TX_POWER_CLOUD = 2.0            # p_cloud: edge-to-cloud TX power (W)

# ============================================================
# Computing Parameters (Table III)
# ============================================================
CPU_FREQ_VEHICLE = 5e8          # f^V: 5 × 10^8 cycles/s (500 MHz)
CPU_FREQ_EDGE = 1e9             # f^E: 1 × 10^9 cycles/s (1 GHz)

# λ: Computation intensity.
# Paper states 10^5 cycles/bit but at d_k=20Mb that gives T_edge=2000s >> Δt=30s.
# Fig 5 shows total delay/episode ≈ 17-22s (sum over 40 slots → ~0.5s avg per slot).
# With T_edge = λ*d_k/f^E, target T_edge ≈ 0.2s → λ = 0.2*f^E/d_k = 10 cycles/bit.
LAMBDA = 10                     # cycles/bit calibrated: T_edge=0.03s << T_cloud=0.3s → edge caching is 10x faster

ALPHA = 2                       # Energy exponent
KAPPA = 1e-26                   # κ: Energy efficiency coefficient

# ============================================================
# Storage & Data Parameters (Table III) — in bits
# ============================================================
DATA_SIZE = 3e6                 # d_k: calibrated data size (bits)
SERVICE_SIZE = 2e6              # θ_k: calibrated Service program storage so NoCache delay ~ 20s
STORAGE_VEHICLE = 2e6          # S^V: Vehicle storage (calibrated to hold 1 service)
STORAGE_EDGE = 4e6            # S^E: Edge storage (calibrated to hold 2 services)

# Vehicle storage can hold at most this many services
MAX_SERVICES_VEHICLE = int(STORAGE_VEHICLE // SERVICE_SIZE)  # = 1
MAX_SERVICES_EDGE = int(STORAGE_EDGE // SERVICE_SIZE)         # = 2

# ============================================================
# Mobility
# ============================================================
V_MIN = 10.0                    # Minimum vehicle speed (m/s)
V_MAX = 30.0                    # Maximum vehicle speed (m/s)
# Each edge covers 500m, 3 edges → 1500m total road
ROAD_LENGTH = NUM_EDGES * EDGE_COMM_RANGE

# ============================================================
# SUMO 2D Mobility Settings
# ============================================================
USE_SUMO = True                         # Toggle 2D trace vs 1D random
SUMO_TRACE_FILE = "car_only/sumo_trace_500.xml"
SUMO_X_MIN = 0.0                        # SUMO network bounding box
SUMO_X_MAX = 10000.0
SUMO_Y_MIN = 0.0
SUMO_Y_MAX = 30000.0

# ============================================================
# DRL Hyperparameters (Table III)
# ============================================================
GAMMA = 0.99                    # γ: Discount factor (Table III)
TAU = 0.01                      # τ: Soft update coefficient (Table III: 0.01)
ACTOR_LR = 0.001                # lr_a: Actor learning rate (Table III: 0.001)
CRITIC_LR = 0.002               # lr_c: Critic learning rate (Table III: 0.002)
REPLAY_SIZE = 10000             # D: Replay buffer capacity (Table III)
BATCH_SIZE = 128                # N: Mini-batch size (Table III: 128)
NOISE_STD_INIT = 0.5            # Initial exploration noise
NOISE_DECAY = 0.98              # Noise decay per episode
HIDDEN_SIZE = 128               # Hidden layer size (actor: 512→256→128→64 hardcoded)

# ============================================================
# Zipf Distribution for Service Popularity
# ============================================================
ZIPF_EXPONENT = 1.2             # Skewness parameter

# ============================================================
# LEAF-specific node capacities (CU = compute units)
# We map CPU cycles/s → CU for LEAF (1 CU = 10^6 cycles/s = 1 MHz)
# ============================================================
VEHICLE_CU = int(CPU_FREQ_VEHICLE / 1e6)   # 500 CU
EDGE_CU = int(CPU_FREQ_EDGE / 1e6)         # 1000 CU
CLOUD_CU = None                             # Unlimited

# LEAF power model parameters for edge servers
EDGE_MAX_POWER = 100.0          # W (total under full load)
EDGE_STATIC_POWER = 20.0        # W (idle)
CLOUD_POWER_PER_CU = 0.0        # Not modelled explicitly (cloud energy tracked separately)
