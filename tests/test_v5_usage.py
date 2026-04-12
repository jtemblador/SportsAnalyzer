# tests/test_v5_usage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.usage import add_usage_features


def test_snap_pct_rolling_uses_prior_weeks():
    """Rolling snap share for week N uses weeks < N only."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'offense_pct': [0.5, 0.7, 0.9],
    })
    out = add_usage_features(df)
    # Week 3 rolling snap pct should be avg of weeks 1-2 = 0.6
    w3 = out[out['week'] == 3].iloc[0]
    assert w3['rolling_offense_pct'] == pytest.approx(0.6, abs=0.01)


def test_snap_trend():
    """prior_week_offense_pct: previous week's value."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'offense_pct': [0.5, 0.7, 0.9],
    })
    out = add_usage_features(df)
    w3 = out[out['week'] == 3].iloc[0]
    assert w3['prior_week_offense_pct'] == pytest.approx(0.7, abs=0.01)


def test_injury_status_encoded():
    """Injury status converted to numeric (Out=3, Doubtful=2, Questionable=1, None=0)."""
    df = pd.DataFrame({
        'player_id': ['A', 'B', 'C', 'D'],
        'player_name': ['A', 'B', 'C', 'D'],
        'position': ['RB']*4, 'season': [2024]*4, 'week': [1]*4,
        'report_status': ['Out', 'Doubtful', 'Questionable', None],
    })
    out = add_usage_features(df)
    assert out.iloc[0]['injury_severity'] == 3
    assert out.iloc[1]['injury_severity'] == 2
    assert out.iloc[2]['injury_severity'] == 1
    assert out.iloc[3]['injury_severity'] == 0


def test_depth_chart_starter_flag():
    """depth_team '1' → is_starter=1, '2' → 0, None → NaN."""
    df = pd.DataFrame({
        'player_id': ['A', 'B', 'C'],
        'player_name': ['A', 'B', 'C'],
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1]*3,
        'depth_team': ['1', '2', None],
    })
    out = add_usage_features(df)
    assert out.iloc[0]['is_starter'] == 1
    assert out.iloc[1]['is_starter'] == 0
    assert pd.isna(out.iloc[2]['is_starter'])
