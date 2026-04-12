# tests/test_v5_rolling.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.rolling import add_rolling_features


@pytest.fixture
def sample_master():
    """4 weeks of data for one player."""
    return pd.DataFrame({
        'player_id': ['X']*4,
        'player_name': ['Test']*4,
        'position': ['QB']*4,
        'season': [2024]*4,
        'week': [1, 2, 3, 4],
        'passing_yards': [200, 250, 300, 150],
        'fantasy_points_ppr': [15.0, 20.0, 25.0, 12.0],
        'carries': [0]*4, 'targets': [0]*4, 'receptions': [0]*4,
        'rushing_yards': [0]*4, 'rushing_tds': [0]*4,
        'receiving_yards': [0]*4, 'receiving_tds': [0]*4,
        'passing_tds': [1, 2, 3, 1], 'passing_interceptions': [0, 0, 1, 1],
    })


def test_rolling_avg_uses_prior_weeks_only(sample_master):
    """Week 3 rolling avg must use weeks 1+2, NOT week 3."""
    df = add_rolling_features(sample_master)
    w3 = df[df['week'] == 3].iloc[0]
    # Chronological prior values for week 3: [15, 20] (w1, w2)
    # Most recent first reversed: [20, 15]
    # Weights with decay 0.85: [1.0, 0.85], sum = 1.85
    # Weighted = (20*1.0 + 15*0.85) / 1.85 = (20 + 12.75) / 1.85 ≈ 17.7
    assert w3['rolling_avg_fantasy_points_ppr'] == pytest.approx(17.7, abs=0.1)


def test_rolling_avg_null_at_week_1(sample_master):
    """Week 1 has no history — rolling features should be NaN."""
    df = add_rolling_features(sample_master)
    w1 = df[df['week'] == 1].iloc[0]
    assert pd.isna(w1['rolling_avg_fantasy_points_ppr'])


def test_variance_computed(sample_master):
    """Variance = std dev of past values."""
    df = add_rolling_features(sample_master)
    w4 = df[df['week'] == 4].iloc[0]
    expected_std = np.std([15.0, 20.0, 25.0])
    assert w4['variance_fantasy_points_ppr'] == pytest.approx(expected_std, abs=0.01)


def test_variance_nan_with_less_than_2_games(sample_master):
    """Week 2 has only 1 prior game — variance should be NaN."""
    df = add_rolling_features(sample_master)
    w2 = df[df['week'] == 2].iloc[0]
    assert pd.isna(w2['variance_fantasy_points_ppr'])


def test_trend_nan_with_less_than_4_games(sample_master):
    """Week 4 has 3 games history — trend needs 4+ to have older window."""
    df = add_rolling_features(sample_master)
    w4 = df[df['week'] == 4].iloc[0]
    assert pd.isna(w4['trend_fantasy_points_ppr'])


def test_no_data_leakage_across_players():
    """One player's rolling avg must not leak into another player's features."""
    df_input = pd.DataFrame({
        'player_id': ['A', 'A', 'B', 'B'],
        'player_name': ['A', 'A', 'B', 'B'],
        'position': ['QB']*4,
        'season': [2024]*4,
        'week': [1, 2, 1, 2],
        'passing_yards': [100, 200, 500, 600],
        'fantasy_points_ppr': [5.0, 10.0, 30.0, 35.0],
        'carries': [0]*4, 'targets': [0]*4, 'receptions': [0]*4,
        'rushing_yards': [0]*4, 'rushing_tds': [0]*4,
        'receiving_yards': [0]*4, 'receiving_tds': [0]*4,
        'passing_tds': [0]*4, 'passing_interceptions': [0]*4,
    })
    df = add_rolling_features(df_input)
    b_w2 = df[(df['player_id'] == 'B') & (df['week'] == 2)].iloc[0]
    # Player B week 2 rolling avg should only use B's week 1 data (30), not A's
    assert b_w2['rolling_avg_fantasy_points_ppr'] == pytest.approx(30.0, abs=0.01)


def test_cross_season_rolling_is_intentional():
    """Week 1 of a new season MUST carry history from the prior season's
    tail games. This is by design — V5_ROADMAP "Season Range: 2018-2025"
    specifies 2018-2019 as warm-up so Week 1 of 2020 has a full lookback.
    The same cross-season behavior applies throughout 2020-2025 training.
    """
    df_input = pd.DataFrame({
        'player_id': ['P'] * 4,
        'player_name': ['Test'] * 4,
        'position': ['QB'] * 4,
        'season': [2023, 2023, 2024, 2024],
        'week': [17, 18, 1, 2],
        'passing_yards': [250, 300, 200, 220],
        'fantasy_points_ppr': [20.0, 25.0, 15.0, 18.0],
        'carries': [0] * 4, 'targets': [0] * 4, 'receptions': [0] * 4,
        'rushing_yards': [0] * 4, 'rushing_tds': [0] * 4,
        'receiving_yards': [0] * 4, 'receiving_tds': [0] * 4,
        'passing_tds': [0] * 4, 'passing_interceptions': [0] * 4,
    })
    df = add_rolling_features(df_input)
    # Week 1 of 2024 MUST carry history from 2023 (not be NaN).
    w1_2024 = df[(df['season'] == 2024) & (df['week'] == 1)].iloc[0]
    assert not pd.isna(w1_2024['rolling_avg_fantasy_points_ppr']), (
        "Week 1 of a new season must have rolling history from prior season"
    )
    # Expect decay-weighted avg of [20.0, 25.0] (chronological), reversed
    # to [25.0, 20.0] with weights [1.0, 0.85]: (25 + 17)/1.85 ≈ 22.7
    assert w1_2024['rolling_avg_fantasy_points_ppr'] == pytest.approx(22.7, abs=0.1)
