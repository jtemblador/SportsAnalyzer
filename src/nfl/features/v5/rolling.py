# src/nfl/features/v5/rolling.py
"""
Rolling average, variance, and trend features.
All use strictly prior weeks (no data leakage into current-week features).
"""

import pandas as pd
from src.nfl.features.v5.config import ROLLING_WINDOW, CORE_STATS_FOR_ROLLING
from src.nfl.features.v5.utils import (
    rolling_decay_avg_series,
    rolling_variance_series,
    rolling_trend_series,
)


def add_rolling_features(df, window=ROLLING_WINDOW, stats=None):
    """
    Add rolling average, variance, and trend features for each core stat.
    Features are computed from weeks STRICTLY prior to current week (no leakage).

    Args:
        df: Master table DataFrame (must be sorted by player_id, season, week)
        window: Number of past games to use for rolling calculations
        stats: List of stat columns to compute features for (default: core stats)

    Returns:
        DataFrame with added columns:
        - rolling_avg_<stat>: decay-weighted average of past N games
        - variance_<stat>: std dev of past games (column name kept for compat)
        - trend_<stat>: recent 3 vs older games percentage change

    Position-safety: uses `groupby(...).transform(...)`, which preserves the
    original DataFrame index regardless of group iteration order. The prior
    list-append + bulk-assign pattern was correct only because the input was
    sorted+reset_index — an unguarded contract. Transform makes that contract
    structural rather than implicit.
    """
    if stats is None:
        stats = [s for s in CORE_STATS_FOR_ROLLING if s in df.columns]

    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    # Add games_of_history column — counts prior games per player across all seasons.
    # Downstream code can use this with MIN_GAMES_HISTORY (config.py) to filter
    # insufficient-history rows or flag rookies for special handling.
    df['games_of_history'] = df.groupby('player_id', sort=False).cumcount()

    # INTENTIONAL: group by player_id only (not [player_id, season]).
    # V5_ROADMAP specifies 2018-2019 as warm-up seasons so that Week 1 of
    # 2020 has a full rolling lookback window from 2019 tail games. This
    # cross-season history is desired throughout training (2020-2025) so
    # that Week 1 of each new season uses the prior season's tail, not
    # empty history. See docs/V5_ROADMAP.md "Season Range: 2018-2025".
    grouped = df.groupby('player_id', sort=False)
    for stat in stats:
        col = grouped[stat]
        df[f'rolling_avg_{stat}'] = col.transform(rolling_decay_avg_series, window=window)
        df[f'variance_{stat}'] = col.transform(rolling_variance_series, window=window)
        df[f'trend_{stat}'] = col.transform(rolling_trend_series, window=window)

    return df
