import pandas as pd

# Load dataset
df = pd.read_csv("tripinfos.csv")

# Remove rows with any missing values
df_clean = df.dropna()

# Save cleaned dataset
df_clean.to_csv("tripinfos_clean.csv", index=False)

print("Original rows:", len(df))
print("Remaining rows after removing missing values:", len(df_clean))
