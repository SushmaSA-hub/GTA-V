# ========== SIMULATION CONTROL ==========
SIM_TIME = 150           # Total simulation time in seconds
NUM_VEHICLES = 250        # Number of autonomous vehicles
NUM_RSUS = 10            # Number of Road Side Units
NUM_BASE_STATIONS = 4    # One per region (matches subregions)
NUM_CHANNELS = 5         # Total channel in the simulation system
ALPHA = 1e-7             # Energy per cycle (local computation)
BETA = 1e-6              # Energy per bit (transmission)


# ========== TASK PARAMETERS ==========
TASK_SIZE_RANGE = (1e6*8, 10e6*8)    # 1-10 MB in bits
TASK_DEADLINE_RANGE = (5, 20)        # Task deadlines in seconds
RESOURCE_PER_BIT = 125               # CPU cycles needed per bit

# ========== VEHICLE PARAMETERS ==========
VEHICLE_SPEED_RANGE = (20, 40)      # km/h
OP_IP_RATIO = 0.1                   # Output/Input ratio
WEIGHTS = [0.5, 0.5]                # Weights for time and energy in cost function

# ========== COMPUTATION FREQUENCIES ==========
FREQ_LOCAL_RANGE = (0.5e9, 0.6e9)   # Vehicle CPU (100-500 MHz)
FREQ_RSU = 10e9                     # RSU CPU (10 GHz)
FREQ_BS = 20e9                      # Base Station CPU (20 GHz)
FREQ_CLOUD = 10e9                   # Cloud CPU per task (10 GHz)

# ========== GEOGRAPHIC PARAMETERS ==========
CENTER_LAT, CENTER_LON = 0, 0       # City center coordinates
AREA_SIDE_KM = 4.0                  # 3km x 3km city area
MAX_OFFSET_KM = AREA_SIDE_KM / 2
MAX_OFFSET_DEG = MAX_OFFSET_KM / 111  # 1° ≈ 111km
SUBREGION_SIDE_DEG = MAX_OFFSET_DEG / 2

# Sub-region centers (4 regions)
SUBREGION_CENTERS = {
    "Bottom-Right": (CENTER_LAT + SUBREGION_SIDE_DEG/2, CENTER_LON - SUBREGION_SIDE_DEG/2),
    "Top-Right": (CENTER_LAT + SUBREGION_SIDE_DEG/2, CENTER_LON + SUBREGION_SIDE_DEG/2),
    "Bottom-Left": (CENTER_LAT - SUBREGION_SIDE_DEG/2, CENTER_LON - SUBREGION_SIDE_DEG/2),
    "Top-Left": (CENTER_LAT - SUBREGION_SIDE_DEG/2, CENTER_LON + SUBREGION_SIDE_DEG/2)
}

# ========== COMMUNICATION PARAMETERS ==========
BANDWIDTH_RANGE = (5e6, 10e6)       # 5-10 MHz bandwidth
TX_POWER = 2                      # Transmission power in watts
NOISE_POWER = 1e-10                 # Noise power in watts
PATH_LOSS_EXP = 4.0                 # Path loss exponent
RSU_RANGE_KM = 0.5                  # 500m RSU coverage

# ========== V2V COOPERATION ==========
V2V_RANGE_KM = 0.3               # Communication range between vehicles (100 meters)
V2V_BANDWIDTH = 5e6              # 5 MHz bandwidth for V2V communication
V2V_TX_POWER = 2                 # Transmission power for V2V (Watts)
