# src/nfl/features/v5/rolling.py
"""
Rolling average, variance, and trend features.
All use strictly prior weeks (no data leakage into current-week features).
"""

import pandas as pd
import numpy as np
from src.nfl.features.v5.config import (
    ROLLING_DECAY, ROLLING_WINDOW, CORE_STATS_FOR_ROLLING
)


def _decay_weighted_avg(values, decay=ROLLING_DECAY):
    """Compute decay-weighted average. Values are in chronological order
    (oldest first). Most-recent values get highest weight."""
    if len(values) == 0:
        return np.nan
    # Reverse so most recent is first, then decay
    rev = values[::-1]
    weights = np.array([decay ** i for i in range(len(rev))])
    return np.sum(rev * weights) / np.sum(weights)


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
        - variance_<stat>: std dev of past games
        - trend_<stat>: recent 3 vs older games percentage change
    """
    if stats is None:
        stats = [s for s in CORE_STATS_FOR_ROLLING if s in df.columns]

    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for stat in stats:
        rolling_avg = []
        variance = []
        trend = []

        for player_id, group in df.groupby('player_id', sort=False):
            values = group[stat].values
            for i in range(len(values)):
                # Use prior N games only (strictly before current)
                past = values[max(0, i - window):i]
                past = past[~pd.isna(past)]

                # Rolling avg (decay-weighted)
                rolling_avg.append(
                    _decay_weighted_avg(past) if len(past) > 0 else np.nan
                )

                # Variance
                variance.append(np.std(past) if len(past) >= 2 else np.nan)

                # Trend: (recent 3 - older) / older
                if len(past) >= 4:
                    recent = np.mean(past[-3:])
                    older = np.mean(past[:-3])
                    trend.append(
                        (recent - older) / older if older != 0 else 0.0
                    )
                else:
                    trend.append(np.nan)

        df[f'rolling_avg_{stat}'] = rolling_avg
        df[f'variance_{stat}'] = variance
        df[f'trend_{stat}'] = trend

    return df
