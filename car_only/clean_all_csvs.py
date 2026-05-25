import pandas as pd
import numpy as np
import glob
import os

# Find all CSV files in current directory
csv_files = glob.glob("*.csv")

print("Found CSV files:")
for f in csv_files:
    print(" -", f)

print("\nCleaning files...\n")

for file in csv_files:
    # Load CSV
    df = pd.read_csv(file)

    original_rows = df.shape[0]

    # Treat empty strings as missing
    df.replace("", np.nan, inplace=True)

    # Drop rows with any missing values
    df_clean = df.dropna(how="any")

    cleaned_rows = df_clean.shape[0]

    # Create new filename
    clean_file = file.replace(".csv", "_clean.csv")

    # Save cleaned CSV
    df_clean.to_csv(clean_file, index=False)

    print(f"{file}:")
    print(f"  Rows before: {original_rows}")
    print(f"  Rows after : {cleaned_rows}")
    print(f"  Removed    : {original_rows - cleaned_rows}")
    print(f"  Saved as   : {clean_file}\n")

print("Cleaning completed ✅")
