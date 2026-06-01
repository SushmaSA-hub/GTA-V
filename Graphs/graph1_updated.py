import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_new = pd.read_csv("C:/Users/Dell/Downloads/GTA-V-main/GTA-V-main/Graphs/Results - Log Total Latency.csv")

print(df_new.head())

plt.figure(figsize=(4,3))
ax = sns.barplot(data=df_new, x="Vehicles count", y="Total Latency (S)", hue="Strategy", errorbar=None)

ax.set_ylim(0, 6)
ax.set_xlabel("Vehicles Count", fontsize=10)
#ax.set_ylabel("Cost (0.7, 0.3)", fontsize=10)
ax.set_ylabel(r'$\log_{10}(Total Latency (S))$', fontsize=10)

ax.legend(['GTA-V', 'DECO','LOCAL', 'RANDOM', 'GREEDY'],
          ncols=3, loc='upper center',
          bbox_to_anchor=(0.5, 1.3),
          frameon=False, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yticks(np.arange(0, 6, 1))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("C:/Users/Dell/Downloads/GTA-V-main/GTA-V-main/Graphs/Log Total Latency.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()