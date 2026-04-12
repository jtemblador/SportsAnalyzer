# src/nfl/features/v5/usage.py
"""
Usage features: snap counts, depth chart, injury status.
"""

import pandas as pd
import numpy as np


# Injury severity mapping
INJURY_SEVERITY = {
    'Out': 3, 'Doubtful': 2, 'Questionable': 1,
    'Probable': 0,  # rarely used, treat as healthy
}


def add_usage_features(df):
    """
    Add snap count rolling features, injury severity, and depth chart flags.
    Uses only prior-week data for snap features (no leakage).
    """
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    # 1. Rolling snap share (prior weeks only).
    # INTENTIONAL cross-season rolling: groupby('player_id') only so Week 1
    # of a new season carries snap-share history from the prior season's
    # tail games (V5_ROADMAP "Season Range: 2018-2025").
    if 'offense_pct' in df.columns:
        df['rolling_offense_pct'] = (
            df.groupby('player_id', sort=False)['offense_pct']
              .apply(lambda s: s.shift(1).expanding().mean())
              .reset_index(level=0, drop=True)
        )

        # Prior week snap pct (for trend detection)
        df['prior_week_offense_pct'] = (
            df.groupby('player_id', sort=False)['offense_pct'].shift(1)
        )

    # 2. Injury severity
    if 'report_status' in df.columns:
        df['injury_severity'] = (
            df['report_status'].map(INJURY_SEVERITY).fillna(0).astype(int)
        )

    # 3. Depth chart starter flag
    if 'depth_team' in df.columns:
        df['is_starter'] = df['depth_team'].apply(
            lambda x: 1 if x == '1' else (0 if pd.notna(x) else np.nan)
        )

    return df
