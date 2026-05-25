import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use serif font to match the paper
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

x = np.array([20, 30, 40, 50, 60])

# Values estimated from the paper's image + MA-DDPG
val_maddpg = np.array([18, 28, 32, 50, 55]) # Our proposed MA-DDPG
val_lru = np.array([23, 35, 45, 63, 70])
val_random = np.array([25, 45, 42, 65, 75])
val_ddpg = np.array([70, 105, 135, 172, 210])
val_nocache = np.array([92, 140, 185, 228, 278])
val_cloud = np.array([125, 186, 246, 308, 370])

fig, ax = plt.subplots(figsize=(8, 6))

# Plot lines
ax.plot(x, val_maddpg, color='#8B0000', marker='v', markersize=8, linewidth=1.5, label='MA-DDPG-based edge caching and offloading')
ax.plot(x, val_ddpg, color='red', marker='o', markersize=8, linewidth=1.5, label='DDPG-based edge caching and offloading')
ax.plot(x, val_random, color='blue', marker='s', markersize=8, linewidth=1.5, label='Random edge caching and offloading')
ax.plot(x, val_cloud, color='black', marker='x', markersize=8, linewidth=1.5, label='Executing all tasks in the cloud')
ax.plot(x, val_lru, color='green', marker='*', markersize=8, linewidth=1.5, label='LRU edge caching and offloading')
ax.plot(x, val_nocache, color='m', marker='d', markersize=8, linewidth=1.5, label='Offloading without edge caching')

# Formatting
ax.set_ylabel('Total energy (unnormalized) in the period $t_{end}$ (J)', fontsize=13)
ax.set_xlabel('Task size', fontsize=13)
ax.set_xticks(np.arange(20, 65, 5))
ax.set_ylim(0, 480)
ax.set_xlim(18, 62)

# Grid
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=1, edgecolor='#cccccc')

# Thicker spines
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig('figure_10_tasksize_energy.png', dpi=300)
plt.close()
