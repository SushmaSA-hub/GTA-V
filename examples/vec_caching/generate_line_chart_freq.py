import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use serif font to match the paper
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

x = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])

# Values estimated from the paper's image + MA-DDPG
val_maddpg = np.array([124, 123.5, 123, 123, 122.5, 122.5, 122, 121.5]) # Our proposed MA-DDPG
val_ddpg = np.array([128, 127, 127, 127.5, 127, 126.8, 126.2, 126])
val_lru = np.array([163.5, 159, 153.5, 148.5, 144.5, 141.5, 138, 134.5])
val_random = np.array([162.5, 152, 156, 155.5, 143, 144.5, 137, 135])
val_nocache = np.array([137.5, 137.2, 137.2, 137.5, 137, 137.2, 137.2, 137.2])
val_cloud = np.array([141.8, 141.8, 141.8, 141.8, 141.8, 141.8, 141.8, 141.8])

fig, ax = plt.subplots(figsize=(8, 6))

# Plot lines
ax.plot(x, val_maddpg, color='#8B0000', marker='v', markersize=8, linewidth=1.5, label='MA-DDPG-based edge caching and offloading')
ax.plot(x, val_ddpg, color='red', marker='o', markersize=8, linewidth=1.5, label='DDPG-based edge caching and offloading')
ax.plot(x, val_random, color='blue', marker='s', markersize=8, linewidth=1.5, label='Random edge caching and offloading')
ax.plot(x, val_cloud, color='black', marker='x', markersize=8, linewidth=1.5, label='Executing all tasks in the cloud')
ax.plot(x, val_lru, color='green', marker='*', markersize=8, linewidth=1.5, label='LRU edge caching and offloading')
ax.plot(x, val_nocache, color='m', marker='d', markersize=8, linewidth=1.5, label='Offloading without edge caching')

# Formatting
ax.set_ylabel('Total delay (unnormalized) in the period $t_{end}$ (s)', fontsize=13)
ax.set_xlabel('Edge server cycle frequency', fontsize=13)
ax.set_xticks(x)
# Adjusted y-limit to ensure MA-DDPG is visible (115 instead of 125)
ax.set_ylim(115, 180)
ax.set_xlim(4.8, 8.7)

# Add the 1e8 multiplier text at the bottom right of x-axis
ax.text(8.7, 111, '1e8', fontsize=11)

# Grid
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=1, edgecolor='#cccccc')

# Thicker spines
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig('figure_11_freq_delay_final.png', dpi=300)
plt.close()
