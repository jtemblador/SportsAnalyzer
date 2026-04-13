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
7. Build and (optionally) save parallel DST team-week feature tables
   as features_dst_{season}.parquet (Task 3.1.5).

Output rows: one per player per week (plus parallel DST team-week tables
when output_dir is provided). Features are all pre-game only.
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
from src.nfl.features.v5.dst import build_dst_features


def build_features(data_dir, seasons, output_dir=None, verbose=True):
    """
    Build V5 features for all players across given seasons — TRAINING DATA.

    This function produces historical feature rows (one per player per
    completed game). It does NOT produce feature rows for upcoming/future
    games — that is the job of a separate inference pipeline (future task).

    The output is intended for model training. For week-ahead predictions
    during the live 2026 season, a separate inference builder will:
      1. Take the upcoming schedule (Vegas lines, weather, opponent)
      2. Look up each player's most recent rolling features from this
         training output
      3. Construct an inference row per (player, upcoming game)
      4. Feed to trained V5 model

    Args:
        data_dir: Path to data/nfl/ (contains subdirs for each dataset)
        seasons: List of season years to process (e.g., [2018, 2019, ..., 2025])
        output_dir: Optional path to save per-season Parquet files.
                    Files will be written to {output_dir}/v5/features_{season}.parquet.
                    If None, only return the in-memory DataFrame.
        verbose: Print progress messages

    Returns:
        DataFrame of features for all (player_id, season, week) combinations.
        NOTE: this is the player-week table only. The parallel DST team-week
        table is produced as a side effect when output_dir is set (written to
        {output_dir}/v5/features_dst_{season}.parquet) and is NOT included in
        the return value. Callers needing DST in-memory should call
        `build_dst_features(data_dir, seasons)` directly.
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
        # Schema consistency note: all per-season outputs below come from the
        # same in-memory DataFrame, so their columns are guaranteed identical
        # within a single build_features() call. Cross-call consistency (e.g.,
        # running once for 2018-2020 and again for 2021-2025 then concatenating)
        # is NOT guaranteed — different season ranges may have different data
        # availability. If that use case arises, add a defensive reindex here
        # against a canonical column list.
        out = Path(output_dir) / VERSION
        out.mkdir(parents=True, exist_ok=True)
        for season, season_df in df.groupby('season'):
            path = out / f'features_{season}.parquet'
            season_df.to_parquet(path)
            if verbose:
                print(f"  Saved {path} ({len(season_df):,} rows)")

        # Parallel DST team-week feature tables (Task 3.1.5).
        if verbose:
            print("Building parallel DST team-week features...")
        build_dst_features(
            data_dir=data_dir,
            seasons=seasons,
            output_dir=output_dir,
            verbose=verbose,
        )

    return df
