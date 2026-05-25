import matplotlib.pyplot as plt

# --- Data Definition ---
# Offloading Method
labels_offload = ['RSU', 'Edge Pool', 'Local']
colors_offload = ['#50ff7b', '#ff6b6b', '#6baed6'] # Green, Red, Blue

ddpg_offload = [75.4, 19.3, 5.3]
maddpg_offload = [68.2, 28.5, 3.3] # MA-DDPG uses edge pool more effectively

# Caching Hit Ratio
labels_cache = ['Cache Hit', 'Cache Miss']
colors_cache = ['#ff0000', '#00ff00'] # Red, Green as in the image

ddpg_cache = [42.4, 57.6]
maddpg_cache = [68.7, 31.3] # MA-DDPG has much higher cache hit ratio


# --- Plotting Configuration ---
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

def make_pie(ax, sizes, labels, colors, title):
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title, fontsize=12, pad=20)

# ==========================================
# FIGURE 3: Offloading Distribution
# ==========================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

make_pie(ax1, ddpg_offload, labels_offload, colors_offload, 'Offloading Method Distribution (DDPG)')
make_pie(ax2, maddpg_offload, labels_offload, colors_offload, 'Offloading Method Distribution (MA-DDPG)')

plt.tight_layout()
fig3.savefig('figure_distribution_1.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# ==========================================
# FIGURE 4: Cache Hit Ratio
# ==========================================
fig4, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

make_pie(ax3, ddpg_cache, labels_cache, colors_cache, 'Cache Hit Ratio (DDPG)')
make_pie(ax4, maddpg_cache, labels_cache, colors_cache, 'Cache Hit Ratio (MA-DDPG)')

plt.tight_layout()
fig4.savefig('figure_distribution_2.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

print("Distribution pie charts generated successfully.")
