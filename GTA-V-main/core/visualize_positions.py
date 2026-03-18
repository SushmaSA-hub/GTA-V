import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Load generated positions
positions_path = os.path.join(os.path.dirname(__file__), "positions.json")
with open(positions_path, "r") as f:
    positions = json.load(f)

# Extract positions
rsus = positions["rsus"]
base_stations = positions["base_stations"]
cloud = positions["cloud"]
vehicles = positions["vehicles"]

# Create plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot parameters
center_lat, center_lon = 0, 0
max_offset = 0.009  # ~1km in degrees

# Draw city boundaries
city_rect = patches.Rectangle(
    (center_lon - max_offset, center_lat - max_offset),
    width=2*max_offset,
    height=2*max_offset,
    linewidth=2,
    edgecolor='black',
    facecolor='none'
)
ax.add_patch(city_rect)

# Plot sub-regions
subregions = [
    ("Bottom-Right", (0.0045, -0.0045)),
    ("Top-Right", (0.0045, 0.0045)),
    ("Bottom-Left", (-0.0045, -0.0045)),
    ("Top-Left", (-0.0045, 0.0045))
]

for name, (clat, clon) in subregions:
    rect = patches.Rectangle(
        (clon - 0.0045, clat - 0.0045),
        width=0.009,
        height=0.009,
        linewidth=1,
        edgecolor='gray',
        facecolor='none',
        linestyle='--'
    )
    ax.add_patch(rect)


# Plot elements
def plot_nodes(nodes, color, marker, label):
    lons = [pos[1] for pos in nodes.values()]
    lats = [pos[0] for pos in nodes.values()]
    ax.scatter(lons, lats, c=color, marker=marker, s=100, label=label)


plot_nodes(rsus, 'pink', 's', 'RSU')
plot_nodes(base_stations, 'deepskyblue', '^', 'Base Station')
plot_nodes(vehicles, 'yellowgreen', 'o', 'Vehicles')
ax.scatter(cloud[1], cloud[0], c='red', marker='*', s=200, label='Cloud')

# Configure plot
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
ax.set_title('Network Node Positions')
ax.grid(True)
ax.legend()
plt.show()
