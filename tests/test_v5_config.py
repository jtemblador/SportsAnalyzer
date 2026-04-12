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
