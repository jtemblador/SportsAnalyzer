import pandas as pd

# Load one file to see the actual column names
df = pd.read_parquet("./data/nfl/raw/player_stats_2025_week_5.parquet")

print("Available columns:")
print(df.columns.tolist())

# Show a few rows to understand the data
print("\nFirst few rows:")
print(df.head())