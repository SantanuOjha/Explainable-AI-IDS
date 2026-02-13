import pandas as pd
import glob
import os

# Directory of THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct absolute path to CSV files
folder_path = os.path.join(BASE_DIR, "..", "Data", "CICIDS_Raw", "*.csv")

# Find CSV files
csv_files = glob.glob(folder_path)

print("CSV files found:", len(csv_files))
print(csv_files[:2])  # preview first 2

# Read and combine
df_list = [pd.read_csv(file, low_memory=False) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

print("Total rows:", combined_df.shape[0])
print("Total columns:", combined_df.shape[1])

# Save merged dataset
output_path = os.path.join(BASE_DIR, "..", "Data", "CICIDS2017_combined.csv")
combined_df.to_csv(output_path, index=False)

print("Merged file saved at:", output_path)
