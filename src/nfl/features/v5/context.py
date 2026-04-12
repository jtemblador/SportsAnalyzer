# src/nfl/features/v5/context.py
"""
Game context features: Vegas lines, weather, rest days, opponent defense rank.
These come from the schedules and team_stats tables.
"""

import pandas as pd
import numpy as np


def add_vegas_features(df):
    """
    Derive Vegas-based features from raw Vegas columns.
    Master table already has: team_implied_total, opponent_implied_total,
    spread_line, total_line, is_home.

    Adds:
    - game_script_index: negative spread (favored) = positive index → pass-heavy
    - total_game_points: the over/under (redundant but named clearly for models)
    """
    if 'spread_line' in df.columns:
        # Game script: favored teams pass less in garbage time, but have
        # more volume early. Negative spread = favored.
        # Index: -spread normalized. Favored by 7 → +7 index.
        df['game_script_index'] = -df['spread_line'].fillna(0)

    if 'total_line' in df.columns:
        df['total_game_points'] = df['total_line']

    return df


def add_weather_features(df):
    """
    Derive weather features. Dome games have NULL temp/wind — treat as
    controlled conditions.

    Adds:
    - is_dome: 1 if roof is dome or closed, 0 otherwise
    - is_high_wind: 1 if wind > 15mph (hurts passing), 0 otherwise
    - is_cold: 1 if temp < 40F, 0 otherwise
    """
    if 'roof' in df.columns:
        df['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
    else:
        df['is_dome'] = 0

    if 'wind' in df.columns:
        df['is_high_wind'] = (df['wind'] > 15).fillna(False).astype(int)
    else:
        df['is_high_wind'] = 0

    if 'temp' in df.columns:
        df['is_cold'] = (df['temp'] < 40).fillna(False).astype(int)
    else:
        df['is_cold'] = 0

    return df


def add_opponent_defense_rank(df):
    """
    Compute opponent defense rank against position based on season-to-date
    fantasy points allowed. Uses only prior weeks (no leakage).

    Adds: opp_def_rank_<position> columns (qb, rb, wr, te)
    """
    df = df.copy()

    for pos in ['QB', 'RB', 'WR', 'TE']:
        col = f'opp_def_rank_{pos.lower()}'
        df[col] = np.nan

    # For each season and each week, compute defense rank using weeks < W
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        weeks = sorted(season_df['week'].unique())

        for week in weeks:
            prior = season_df[season_df['week'] < week]

            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_prior = prior[prior['position'] == pos]
                if pos_prior.empty:
                    continue

                # Avg fantasy points allowed per defense
                avg_allowed = pos_prior.groupby('opponent_team')[
                    'fantasy_points_ppr'
                ].mean()

                # Rank 1 = toughest (allows fewest), 32 = easiest
                ranks = avg_allowed.rank(method='min')

                col = f'opp_def_rank_{pos.lower()}'
                mask = (df['season'] == season) & (df['week'] == week)
                for def_team, rank in ranks.items():
                    df.loc[mask & (df['opponent_team'] == def_team), col] = rank

    return df
