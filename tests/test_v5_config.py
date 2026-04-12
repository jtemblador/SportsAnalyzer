# tests/test_v5_config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nfl.features.v5.config import (
    VERSION,
    FEATURE_GROUPS,
    STATS_TO_PREDICT,
    ROLLING_DECAY,
    MIN_GAMES_HISTORY,
)


def test_version_string():
    assert VERSION == 'v5'


def test_feature_groups_defined():
    expected = ['rolling', 'context', 'usage', 'advanced']
    for group in expected:
        assert group in FEATURE_GROUPS


def test_stats_to_predict_per_position():
    assert 'passing_yards' in STATS_TO_PREDICT['QB']
    assert 'passing_tds' in STATS_TO_PREDICT['QB']
    assert 'rushing_yards' in STATS_TO_PREDICT['RB']
    assert 'receiving_yards' in STATS_TO_PREDICT['WR']
    assert 'receiving_yards' in STATS_TO_PREDICT['TE']
    assert 'fg_made' in STATS_TO_PREDICT['K']


def test_rolling_decay_is_v4_proven_value():
    """V4 used 0.85 decay (proven to work). Keep it."""
    assert ROLLING_DECAY == 0.85


def test_min_games_history():
    """3-game minimum per V5_QUESTIONS.md decision."""
    assert MIN_GAMES_HISTORY == 3


def test_feature_group_columns_by_group():
    """Verify group column mapping works for each group."""
    from src.nfl.features.v5.config import get_feature_columns_by_group

    cols = [
        'player_id', 'player_name',  # metadata
        'rolling_avg_fantasy_points_ppr', 'variance_passing_yards', 'trend_rushing_yards',  # rolling
        'game_script_index', 'is_dome', 'opp_def_rank_qb',  # context
        'injury_severity', 'is_starter', 'rolling_offense_pct',  # usage
        'rolling_ngs_passing_avg_time_to_throw', 'rolling_pfr_pass_times_pressured_pct',  # advanced
        'games_of_history',  # rolling
    ]

    rolling = get_feature_columns_by_group(cols, 'rolling')
    assert 'rolling_avg_fantasy_points_ppr' in rolling
    assert 'variance_passing_yards' in rolling
    assert 'trend_rushing_yards' in rolling
    assert 'games_of_history' in rolling
    assert 'game_script_index' not in rolling

    context = get_feature_columns_by_group(cols, 'context')
    assert 'game_script_index' in context
    assert 'is_dome' in context
    assert 'opp_def_rank_qb' in context

    usage = get_feature_columns_by_group(cols, 'usage')
    assert 'injury_severity' in usage
    assert 'is_starter' in usage
    assert 'rolling_offense_pct' in usage

    advanced = get_feature_columns_by_group(cols, 'advanced')
    assert 'rolling_ngs_passing_avg_time_to_throw' in advanced
    assert 'rolling_pfr_pass_times_pressured_pct' in advanced
