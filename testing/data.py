import pandas as pd

# Read a feature file
df = pd.read_parquet('./data/nfl/cleaned/features_2025_week_9.parquet')
print(df.columns)  # See all calculated features
print(df.head())   # See first few rows