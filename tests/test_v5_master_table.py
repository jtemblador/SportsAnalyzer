# tests/test_v5_master_table.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.nfl.features.v5.master_table import build_master_table

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'v5_mini_data'


@pytest.fixture
def master_df():
    return build_master_table(data_dir=str(FIXTURE_DIR), seasons=[2024])


def test_has_one_row_per_player_week(master_df):
    # 2 players × 3 weeks = 6 rows
    assert len(master_df) == 6


def test_preserves_weekly_stats_columns(master_df):
    assert 'fantasy_points_ppr' in master_df.columns
    assert 'passing_yards' in master_df.columns


def test_joins_players_for_pfr_id(master_df):
    mahomes = master_df[master_df['player_id'] == '00-0033873'].iloc[0]
    assert mahomes['pfr_id'] == 'MahoPa00'


def test_joins_snap_counts_via_pfr(master_df):
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    assert mahomes_w1['offense_pct'] == 1.0


def test_joins_injuries_nullable(master_df):
    # Mahomes W3 was Questionable (has injury row)
    mahomes_w3 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 3)
    ].iloc[0]
    assert mahomes_w3['report_status'] == 'Questionable'

    # Mahomes W1 has no injury row — should be NULL
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    assert pd.isna(mahomes_w1['report_status'])


def test_joins_schedules_game_context(master_df):
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    # KC hosted BAL week 1, spread was 3.0
    assert mahomes_w1['spread_line'] == 3.0
    assert mahomes_w1['total_line'] == 46.0
