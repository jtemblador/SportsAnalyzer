# src/nfl/features/v5/engineer.py
"""
V5 feature engineering orchestrator.

Entry point: build_features(data_dir, seasons, output_dir=None)

Pipeline:
1. Build master player-week table from all 13 Parquet datasets
2. Add rolling features (averages, variance, trends)
3. Add context features (Vegas, weather, opponent rank)
4. Add usage features (snap counts, injury, depth)
5. Add advanced features (NGS, PFR, FF opportunity rolling)
6. Optionally save per-season Parquet files

Output rows: one per player per week. Features are all pre-game only.
"""

import pandas as pd
from pathlib import Path

from src.nfl.features.v5.config import VERSION
from src.nfl.features.v5.master_table import build_master_table
from src.nfl.features.v5.rolling import add_rolling_features
from src.nfl.features.v5.context import (
    add_vegas_features, add_weather_features, add_opponent_defense_rank
)
from src.nfl.features.v5.usage import add_usage_features
from src.nfl.features.v5.advanced import add_advanced_features


def build_features(data_dir, seasons, output_dir=None, verbose=True):
    """
    Build V5 features for all players across given seasons.

    Args:
        data_dir: Path to data/nfl/ (contains subdirs for each dataset)
        seasons: List of season years to process (e.g., [2018, 2019, ..., 2025])
        output_dir: Optional path to save per-season Parquet files.
                    Files will be written to {output_dir}/v5/features_{season}.parquet.
                    If None, only return the in-memory DataFrame.
        verbose: Print progress messages

    Returns:
        DataFrame of features for all (player_id, season, week) combinations.
    """
    if verbose:
        print(f"V5 feature engineering: seasons {min(seasons)}-{max(seasons)}")
        print(f"Loading master player-week table...")

    df = build_master_table(data_dir=data_dir, seasons=seasons)
    if verbose:
        print(f"  Master table: {len(df):,} rows, {len(df.columns)} columns")

    if verbose:
        print("Computing rolling features (averages, variance, trends)...")
    df = add_rolling_features(df)

    if verbose:
        print("Computing context features (Vegas, weather, opponent rank)...")
    df = add_vegas_features(df)
    df = add_weather_features(df)
    df = add_opponent_defense_rank(df)

    if verbose:
        print("Computing usage features (snap, injury, depth chart)...")
    df = add_usage_features(df)

    if verbose:
        print("Computing advanced features (NGS, PFR, FF opp)...")
    df = add_advanced_features(df)

    if verbose:
        print(f"Final feature table: {len(df):,} rows, {len(df.columns)} columns")

    if output_dir:
        out = Path(output_dir) / VERSION
        out.mkdir(parents=True, exist_ok=True)
        for season, season_df in df.groupby('season'):
            path = out / f'features_{season}.parquet'
            season_df.to_parquet(path)
            if verbose:
                print(f"  Saved {path} ({len(season_df):,} rows)")

    return df
