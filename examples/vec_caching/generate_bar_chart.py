import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use serif font to match the paper
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

densities = np.array([2, 3, 4, 5])
x = np.arange(len(densities))

# Values estimated from the paper's image
val_maddpg = np.array([13.8, 16.0, 18.2, 19.4]) # Our proposed MA-DDPG (better than DDPG)
val_ddpg = np.array([14.8, 17.2, 19.2, 20.6])
val_lru = np.array([16.1, 18.3, 20.6, 21.4])
val_random = np.array([16.8, 19.1, 21.2, 21.8])
val_cloud = np.array([16.9, 19.5, 22.1, 23.3])
val_nocache = np.array([18.3, 20.0, 21.8, 22.5])

width = 0.12 # Width of the bars
offset = width

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
ax.bar(x - 2.5*offset, val_maddpg, width, label='MA-DDPG-based edge caching and offloading', color='#8B0000', edgecolor='white', linewidth=0.5)
ax.bar(x - 1.5*offset, val_ddpg, width, label='DDPG-based edge caching and offloading', color='#E6292B', edgecolor='white', linewidth=0.5)
ax.bar(x - 0.5*offset, val_lru, width, label='LRU edge caching and offloading', color='#2EA144', edgecolor='white', linewidth=0.5)
ax.bar(x + 0.5*offset, val_random, width, label='Random edge caching and offloading', color='#1D7AC3', edgecolor='white', linewidth=0.5)
ax.bar(x + 1.5*offset, val_cloud, width, label='Executing all tasks in the cloud', color='#F7931E', edgecolor='white', linewidth=0.5)
ax.bar(x + 2.5*offset, val_nocache, width, label='Offloading without edge caching', color='#6E489E', edgecolor='white', linewidth=0.5)

# Formatting
ax.set_ylabel('Total delay in the period $t_{end}$ (s)', fontsize=14)
ax.set_xlabel('The vehicle density at each edge', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(densities, fontsize=12)
ax.set_ylim(0, 40)

# Y-axis ticks from 0 to 40 with step 5
ax.set_yticks(np.arange(0, 41, 5))

# Add a dashed grid for the y-axis
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
ax.set_axisbelow(True) # Put grid behind bars

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=1, borderpad=0.8, edgecolor='#cccccc')

# Set figure borders slightly thicker to match paper
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig('figure_7_density_delay.png', dpi=300)
plt.close()
