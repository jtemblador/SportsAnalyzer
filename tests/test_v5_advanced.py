# tests/test_v5_advanced.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.advanced import add_advanced_features


def test_ngs_rolling_passing():
    """NGS time_to_throw rolling average uses prior weeks."""
    df = pd.DataFrame({
        'player_id': ['QB1']*3, 'player_name': ['QB1']*3,
        'position': ['QB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'ngs_passing_avg_time_to_throw': [2.5, 2.7, 2.6],
        'ngs_passing_completion_percentage_above_expectation': [1.0, 2.0, 0.5],
    })
    out = add_advanced_features(df)
    w3 = out[out['week'] == 3].iloc[0]
    # Rolling avg of weeks 1-2: (2.5 + 2.7) / 2 = 2.6
    assert w3['rolling_ngs_passing_avg_time_to_throw'] == pytest.approx(2.6, abs=0.01)


def test_ff_opp_features():
    """FF opportunity: rolling expected fantasy points + differential."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'total_fantasy_points_exp': [15.0, 18.0, 12.0],
        'total_fantasy_points_diff': [2.0, -3.0, 5.0],
    })
    out = add_advanced_features(df)
    w3 = out[out['week'] == 3].iloc[0]
    # Rolling avg exp (weeks 1-2) = 16.5
    assert w3['rolling_total_fantasy_points_exp'] == pytest.approx(16.5, abs=0.01)
    # Rolling avg diff (weeks 1-2) = -0.5
    assert w3['rolling_total_fantasy_points_diff'] == pytest.approx(-0.5, abs=0.01)


def test_null_preserved_for_unqualified_players():
    """Non-qualified players have NULL NGS — preserve NULL."""
    df = pd.DataFrame({
        'player_id': ['A']*2, 'player_name': ['A']*2,
        'position': ['RB']*2, 'season': [2024]*2, 'week': [1, 2],
        'ngs_rushing_efficiency': [None, None],
    })
    out = add_advanced_features(df)
    w2 = out[out['week'] == 2].iloc[0]
    assert pd.isna(w2['rolling_ngs_rushing_efficiency'])
