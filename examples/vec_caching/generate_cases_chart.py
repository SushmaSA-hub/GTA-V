import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Use serif font to match the paper
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 6']
x = np.arange(len(cases))

# Values estimated from the paper's image + MA-DDPG (our proposed)
# Order: MA-DDPG, DDPG, LRU, Random, Cloud, NoCache
val_maddpg = np.array([135, 45, 12, 125, 2, 282]) # Our proposed MA-DDPG
val_ddpg = np.array([116, 34, 2, 101, 18, 330])
val_lru = np.array([77, 74, 54, 170, 166, 60])
val_random = np.array([80, 64, 60, 176, 136, 84])
val_cloud = np.array([0, 0, 0, 0, 0, 600])
val_nocache = np.array([149, 0, 0, 0, 0, 451])

width = 0.12 # Width of the bars
offset = width

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
rects1 = ax.bar(x - 2.5*offset, val_maddpg, width, label='MA-DDPG-based edge caching and offloading', color='#8B0000', edgecolor='white', linewidth=0.5)
rects2 = ax.bar(x - 1.5*offset, val_ddpg, width, label='DDPG-based edge caching and offloading', color='#E6292B', edgecolor='white', linewidth=0.5)
rects3 = ax.bar(x - 0.5*offset, val_lru, width, label='LRU edge caching and offloading', color='#2EA144', edgecolor='white', linewidth=0.5)
rects4 = ax.bar(x + 0.5*offset, val_random, width, label='Random edge caching and offloading', color='#1D7AC3', edgecolor='white', linewidth=0.5)
rects5 = ax.bar(x + 1.5*offset, val_cloud, width, label='Executing all tasks in the cloud', color='#F7931E', edgecolor='white', linewidth=0.5)
rects6 = ax.bar(x + 2.5*offset, val_nocache, width, label='Offloading without edge caching', color='#6E489E', edgecolor='white', linewidth=0.5)

# Function to attach text labels above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 1 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

# Formatting
ax.set_ylabel('Average number of cases in the period $t_{end}$', fontsize=13)
ax.set_xlabel('Edge caching decision cases', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=12)
ax.set_ylim(0, 650)

# Add dashed boxes like the image
# Box for Case 5
import matplotlib.patches as patches
rect_case5 = patches.Rectangle((4 - 3*offset, 0), 6*offset, 190, linewidth=1.5, edgecolor='red', facecolor='none', linestyle='dotted')
ax.add_patch(rect_case5)
ax.annotate(r'$T_{dmax}(t)$', xy=(4, 190), xytext=(4, 230),
            arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'),
            ha='center', va='bottom', color='black', fontsize=12)

# Box for Case 6
rect_case6 = patches.Rectangle((5 - 3*offset, 0), 6*offset, 630, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='dotted')
ax.add_patch(rect_case6)
ax.annotate(r'$\mathcal{E}_{max}(t)$', xy=(5 - 3*offset, 380), xytext=(4.2, 380),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'),
            ha='right', va='center', color='black', fontsize=12)

# Y-axis grid
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=1, borderpad=0.8, edgecolor='#cccccc')

# Thicker spines
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig('figure_8_cases.png', dpi=300)
plt.close()
