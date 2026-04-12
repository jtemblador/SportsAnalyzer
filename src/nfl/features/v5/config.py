# src/nfl/features/v5/config.py
"""
V5 feature engineering configuration.
Defines stats to predict per position, feature groups, and tuning constants.
"""

VERSION = 'v5'

# Rolling average decay factor — proven in V4 (stronger emphasis on recent 3 games)
ROLLING_DECAY = 0.85

# Minimum games of history required to generate predictions
# Below this, output 'insufficient_data' flag instead of predictions
MIN_GAMES_HISTORY = 3

# Number of past games to use for rolling calculations
ROLLING_WINDOW = 6

# Stats to predict per position (V5_QUESTIONS.md decision)
STATS_TO_PREDICT = {
    'QB': ['passing_yards', 'passing_tds', 'passing_interceptions',
           'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions',
           'receiving_yards', 'receiving_tds'],
    'WR': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
    'TE': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
    'K':  ['fg_made', 'fg_att', 'pat_made'],
}

# Feature groups for ablation study (Task 3.2b will remove one at a time)
FEATURE_GROUPS = {
    'rolling':   'Rolling averages, variance, and trend features (V2/V4 proven)',
    'context':   'Vegas lines, weather, rest, opponent defense rank (V4 proven)',
    'usage':     'Snap counts, depth chart status, injury reports',
    'advanced':  'NGS metrics, PFR advanced stats, FF opportunity (lower confidence)',
}

# Core stats used for rolling average features (all positions get these)
CORE_STATS_FOR_ROLLING = [
    'fantasy_points_ppr',
    'passing_yards', 'passing_tds', 'passing_interceptions',
    'rushing_yards', 'rushing_tds', 'carries',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
    # Kicker stats — required so kicker predictions have rolling features
    'fg_made', 'fg_att', 'pat_made',
]

# Fantasy PPR scoring formula — used to derive fantasy points from predicted stats
FANTASY_PPR_WEIGHTS = {
    'passing_yards': 0.04,
    'passing_tds': 4.0,
    'passing_interceptions': -2.0,
    'rushing_yards': 0.1,
    'rushing_tds': 6.0,
    'receptions': 1.0,
    'receiving_yards': 0.1,
    'receiving_tds': 6.0,
}


# Column prefixes/exact names that belong to each feature group.
# Used by Task 3.2b ablation study to drop one group at a time.
# A feature column belongs to group G if it starts with any prefix in
# FEATURE_GROUP_PREFIXES[G]['prefixes'] OR is in the 'exact' list.
FEATURE_GROUP_PREFIXES = {
    'rolling': {
        'prefixes': ['rolling_avg_', 'variance_', 'trend_'],
        'exact': ['games_of_history'],
    },
    'context': {
        'prefixes': ['opp_def_rank_'],
        'exact': [
            'game_script_index', 'total_game_points',
            'is_dome', 'is_high_wind', 'is_cold', 'is_home',
            'team_implied_total', 'opponent_implied_total',
            'spread_line', 'total_line', 'team_rest', 'opponent_rest',
            'temp', 'wind', 'roof', 'div_game',
        ],
    },
    'usage': {
        'prefixes': ['rolling_offense_pct', 'prior_week_offense_pct'],
        'exact': ['injury_severity', 'is_starter'],
    },
    'advanced': {
        'prefixes': ['rolling_ngs_', 'rolling_pfr_', 'rolling_total_fantasy_points_',
                     'rolling_pass_fantasy_points_', 'rolling_rec_fantasy_points_',
                     'rolling_rush_fantasy_points_'],
        'exact': [],
    },
}


def get_feature_columns_by_group(df_columns, group):
    """
    Given a list of column names and a feature group name, return the subset
    of columns belonging to that group.

    Args:
        df_columns: list of column names from a feature DataFrame
        group: one of 'rolling', 'context', 'usage', 'advanced'

    Returns:
        List of column names that belong to the specified group.
    """
    if group not in FEATURE_GROUP_PREFIXES:
        raise ValueError(
            f"Unknown group '{group}'. Must be one of {list(FEATURE_GROUP_PREFIXES)}"
        )

    spec = FEATURE_GROUP_PREFIXES[group]
    prefixes = spec['prefixes']
    exacts = set(spec['exact'])

    matches = []
    for col in df_columns:
        if col in exacts:
            matches.append(col)
        elif any(col.startswith(p) for p in prefixes):
            matches.append(col)
    return matches
