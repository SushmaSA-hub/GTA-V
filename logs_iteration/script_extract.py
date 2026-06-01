import pandas as pd

# Replace 'data.csv' with your actual CSV file path
df = pd.read_csv('simulation_results_summary_1749905462.csv')

# Filter for rows where strategy is 'Proposed'
filtered_df = df[df['strategy'] == 'Local']

# Save the filtered data to a text file in table format
filtered_df.to_csv('local.txt', index=False, sep='\t')
