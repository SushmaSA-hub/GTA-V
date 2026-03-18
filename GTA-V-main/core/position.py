import os
import random
import math
from .parameters import *


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (in kilometers)"""
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_rsu_positions() -> list:
    """Generate RSU positions with minimum distance constraints"""
    positions = []
    while len(positions) < NUM_RSUS:
        lat = random.uniform(CENTER_LAT - MAX_OFFSET_DEG, CENTER_LAT + MAX_OFFSET_DEG)
        lon = random.uniform(CENTER_LON - MAX_OFFSET_DEG, CENTER_LON + MAX_OFFSET_DEG)
        if all(haversine(lat, lon, p[0], p[1]) >= RSU_RANGE_KM for p in positions):
            positions.append((lat, lon))
    return positions


def generate_vehicle_positions() -> list:
    """Generate vehicle positions within random sub-regions"""
    positions = []
    for _ in range(NUM_VEHICLES):
        region = random.choice(list(SUBREGION_CENTERS.values()))
        lat = random.uniform(region[0] - SUBREGION_SIDE_DEG / 2,
                             region[0] + SUBREGION_SIDE_DEG / 2)
        lon = random.uniform(region[1] - SUBREGION_SIDE_DEG / 2,
                             region[1] + SUBREGION_SIDE_DEG / 2)
        positions.append((lat, lon))
    return positions


# Generated positions
rsu_positions = generate_rsu_positions()
# base_station_positions = list(SUBREGION_CENTERS.values())  # One per region
base_station_positions = []
for region, center in SUBREGION_CENTERS.items():
    lat = random.uniform(center[0] - SUBREGION_SIDE_DEG/2, center[0] + SUBREGION_SIDE_DEG/2)
    lon = random.uniform(center[1] - SUBREGION_SIDE_DEG/2, center[1] + SUBREGION_SIDE_DEG/2)
    base_station_positions.append((lat, lon))
cloud_position = (CENTER_LAT + 1.3, CENTER_LON + 1.3)  # 144km away from center
vehicle_positions = generate_vehicle_positions()


def export_positions():
    """Export positions to JSON for visualization"""
    import json
    data = {
        "rsus": {f"RSU{i}": pos for i, pos in enumerate(rsu_positions)},
        "base_stations": {f"BS{i}": pos for i, pos in enumerate(base_station_positions)},
        "cloud": cloud_position,
        "vehicles": {f"V{i}": pos for i, pos in enumerate(vehicle_positions)}
    }
    positions_path = os.path.join(os.path.dirname(__file__), 'positions.json')
    with open(positions_path, "w") as f:
        json.dump(data, f, indent=2)


# Create position file when module loads
export_positions()

# ... (your existing code)

if __name__ == "__main__":
    # Generate positions when this file is run directly
    export_positions()
    print("positions.json generated successfully!")
