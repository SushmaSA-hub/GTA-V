import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: make style close to the reference figure
sns.set_style("whitegrid")

df_new = pd.read_csv(
    "C:/Users/Dell/Downloads/GTA-V-main/GTA-V-main/Graphs/Results - satisfaction_ratio.csv"
)

print(df_new.head())

# Color mapping taken to visually match satisfaction_ratio-1.pdf
# GTA-V  -> Proposed
# DECO   -> DECO
# RANDOM -> Random
# GREEDY -> Greedy
palette_map = {
    "GTA-V":  "#72B6A1",   # teal/green
    "DECO":   "#E99675",   # orange/salmon
    "RANDOM": "#D69BC4",   # pink/lilac
    "GREEDY": "#95BC5A"    # olive green
}

# Keep hue order fixed so legend and colors stay consistent
hue_order = ["GTA-V", "DECO", "RANDOM", "GREEDY"]

plt.figure(figsize=(4, 3))

ax = sns.barplot(
    data=df_new,
    x="Vehicles count",
    y="Satisfaction Ratio",
    hue="Strategy",
    hue_order=hue_order,
    palette=palette_map,
    errorbar=None
)

ax.set_ylim(0, 70)
ax.set_xlabel("Vehicles Count", fontsize=10)
ax.set_ylabel("Satisfaction Ratio (%)", fontsize=10)

ax.legend(
    title=None,
    ncols=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.3),
    frameon=False,
    fontsize=10
)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yticks(np.arange(0, 80, 10))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig(
    "C:/Users/Dell/Downloads/GTA-V-main/GTA-V-main/Graphs/sratio.pdf",
    format='pdf',
    dpi=300,
    bbox_inches='tight'
)
plt.show()