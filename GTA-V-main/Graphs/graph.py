import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("Results - satisfaction_ratio.csv")

# Convert columns if needed
df["Vehicles count"] = df["Vehicles count"].astype(str)  # Keep as string for grouping in plots

# Set plot style
sns.set(style="whitegrid")

# Create a figure and axes
plt.figure(figsize=(8, 6))

# Boxplot (a.k.a. candle plot)
ax = sns.barplot(
    data=df,
    x="Vehicles count",            # X-axis: different vehicle counts
    y="Satisfaction Ratio",        # Y-axis: the metric being measured
    hue="Strategy",                # Hue: different strategies
    palette="Set2",                # Optional: color palette
    linewidth=1.5
)

# Labels and Title
# ax.set_title("Satisfaction Ratio Distribution by Strategy", fontsize=14)
ax.set_xlabel("Vehicles Count", fontsize=17)
# ax.set_ylabel(r'$\log_{10}(\mathrm{Cost (0.3, 0.7)})$', fontsize=17)
ax.set_ylabel("Satisfaction Ratio(%)", fontsize=17)
# Customize legend
ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=4, frameon=False, fontsize=17)

# Aesthetic tweaks
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Save or show
plt.savefig("satisfaction.pdf", format='pdf', dpi=300)
plt.show()
