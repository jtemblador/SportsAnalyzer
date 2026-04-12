# tests/test_v5_engineer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.nfl.features.v5.engineer import build_features

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'v5_mini_data'


def test_build_features_returns_dataframe():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_has_rolling_features():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'rolling_avg_fantasy_points_ppr' in df.columns


def test_has_context_features():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'game_script_index' in df.columns
    assert 'is_dome' in df.columns


def test_has_usage_features():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'injury_severity' in df.columns


def test_saves_per_season_parquet(tmp_path):
    """build_features with output_dir should save per-season Parquet files."""
    df = build_features(
        data_dir=str(FIXTURE_DIR), seasons=[2024],
        output_dir=str(tmp_path),
    )
    expected_file = tmp_path / 'v5' / 'features_2024.parquet'
    assert expected_file.exists()


def test_feature_count_reasonable():
    """V5 should produce 30+ feature columns on minimal fixture."""
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    feature_cols = [
        c for c in df.columns
        if c.startswith(('rolling_', 'variance_', 'trend_'))
        or c in ['game_script_index', 'is_dome', 'is_high_wind', 'is_cold',
                 'injury_severity', 'is_starter', 'is_home']
        or c.startswith('opp_def_rank_')
    ]
    assert len(feature_cols) >= 30, f"Only {len(feature_cols)} features"
