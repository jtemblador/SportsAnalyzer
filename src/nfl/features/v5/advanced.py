# src/nfl/features/v5/advanced.py
"""
Advanced features: NGS, PFR, FF opportunity.
All are rolling averages of prior-week advanced metrics.
"""

import pandas as pd
import numpy as np


# NGS columns to compute rolling features on
NGS_COLUMNS = [
    'ngs_passing_avg_time_to_throw',
    'ngs_passing_completion_percentage_above_expectation',
    'ngs_passing_aggressiveness',
    'ngs_rushing_efficiency',
    'ngs_rushing_rush_yards_over_expected_per_att',
    'ngs_rushing_percent_attempts_gte_eight_defenders',
    'ngs_receiving_avg_separation',
    'ngs_receiving_avg_yac_above_expectation',
    'ngs_receiving_avg_cushion',
    'ngs_receiving_catch_percentage',
]

# PFR columns to compute rolling features on
PFR_COLUMNS = [
    'pfr_pass_times_pressured_pct',
    'pfr_pass_passing_bad_throw_pct',
    'pfr_pass_passing_drops',
    'pfr_rush_rushing_yards_after_contact_avg',
    'pfr_rush_rushing_broken_tackles',
    'pfr_rec_receiving_drop_pct',
    'pfr_rec_receiving_broken_tackles',
    'pfr_rec_receiving_rat',
]

# FF opportunity columns
FF_OPP_COLUMNS = [
    'total_fantasy_points_exp',
    'total_fantasy_points_diff',
    'pass_fantasy_points_exp',
    'rec_fantasy_points_exp',
    'rush_fantasy_points_exp',
]


def _rolling_prior_mean(df, col):
    """Rolling mean using prior weeks only (strict shift-then-expand)."""
    return (
        df.groupby('player_id', sort=False)[col]
          .apply(lambda s: s.shift(1).expanding().mean())
          .reset_index(level=0, drop=True)
    )


def add_advanced_features(df):
    """Add rolling averages for NGS, PFR, and FF opportunity columns."""
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for col in NGS_COLUMNS + PFR_COLUMNS + FF_OPP_COLUMNS:
        if col in df.columns:
            df[f'rolling_{col}'] = _rolling_prior_mean(df, col)

    return df
