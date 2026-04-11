"""
Tests for the database query layer (src/nfl/db/queries.py).
Verifies all 7 query functions against live data with known players/games.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.db.queries import (
    get_player_history,
    get_week_stats,
    get_player_injuries,
    get_snap_share,
    get_game_context,
    get_opponent_defense_rank,
    get_nextgen_stats,
)

# Known player IDs (GSIS)
MAHOMES_ID = '00-0033873'
BARKLEY_ID = '00-0034844'


class TestGetPlayerHistory:

    def test_returns_correct_number_of_games(self):
        df = get_player_history(MAHOMES_ID, 2024, 10, games_back=6)
        assert len(df) == 6

    def test_all_rows_before_target_week(self):
        df = get_player_history(MAHOMES_ID, 2024, 5, games_back=6)
        for _, row in df.iterrows():
            assert (row['season'] < 2024) or (row['season'] == 2024 and row['week'] < 5)

    def test_cross_season_lookback(self):
        """Week 2 of 2024 should pull games from 2023."""
        df = get_player_history(MAHOMES_ID, 2024, 2, games_back=6)
        assert len(df) == 6
        seasons = df['season'].unique()
        assert 2023 in seasons

    def test_ordered_most_recent_first(self):
        df = get_player_history(MAHOMES_ID, 2024, 10, games_back=6)
        for i in range(len(df) - 1):
            current = (df.iloc[i]['season'], df.iloc[i]['week'])
            next_row = (df.iloc[i+1]['season'], df.iloc[i+1]['week'])
            assert current >= next_row

    def test_nonexistent_player_returns_empty(self):
        df = get_player_history('FAKE-ID-12345', 2024, 5)
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_mahomes_stats_reasonable(self):
        df = get_player_history(MAHOMES_ID, 2024, 5, games_back=4)
        assert df['passing_yards'].mean() > 100
        assert df['fantasy_points_ppr'].mean() > 5


class TestGetWeekStats:

    def test_returns_data(self):
        df = get_week_stats(2024, 1)
        assert len(df) > 50

    def test_position_filter(self):
        df = get_week_stats(2024, 1, position='QB')
        assert len(df) > 10
        assert (df['position'] == 'QB').all()

    def test_no_position_returns_all(self):
        df_all = get_week_stats(2024, 1)
        df_qb = get_week_stats(2024, 1, position='QB')
        assert len(df_all) > len(df_qb)


class TestGetPlayerInjuries:

    def test_mahomes_w18_doubtful(self):
        """Mahomes was Doubtful with ankle injury in 2024 W18."""
        df = get_player_injuries(MAHOMES_ID, 2024, 18)
        assert len(df) >= 1
        assert df.iloc[0]['report_status'] == 'Doubtful'

    def test_healthy_player_returns_empty(self):
        """Mahomes W1 2024 — not on injury report."""
        df = get_player_injuries(MAHOMES_ID, 2024, 1)
        assert len(df) == 0

    def test_returns_dataframe(self):
        df = get_player_injuries('FAKE-ID', 2024, 1)
        assert isinstance(df, pd.DataFrame)


class TestGetSnapShare:

    def test_barkley_w1_starter(self):
        """Barkley should have ~80% offense snaps in 2024 W1."""
        df = get_snap_share(BARKLEY_ID, 2024, 1)
        assert len(df) == 1
        assert df.iloc[0]['offense_pct'] >= 0.5

    def test_gsis_to_pfr_mapping_works(self):
        """Using GSIS ID should find PFR-keyed snap data."""
        df = get_snap_share(MAHOMES_ID, 2024, 1)
        assert len(df) == 1
        assert df.iloc[0]['offense_pct'] > 0.9  # QB plays almost every snap

    def test_unmapped_player_returns_empty(self):
        df = get_snap_share('FAKE-ID-12345', 2024, 1)
        assert len(df) == 0


class TestGetGameContext:

    def test_kc_w1_2024(self):
        """KC hosted BAL in 2024 W1: spread 3.0, total 46.0."""
        df = get_game_context(2024, 1, 'KC')
        assert len(df) == 1
        row = df.iloc[0]
        assert row['home_team'] == 'KC'
        assert row['away_team'] == 'BAL'
        assert row['spread_line'] == 3.0
        assert row['total_line'] == 46.0
        assert row['is_home'] == True

    def test_away_team_context(self):
        """BAL was away at KC in 2024 W1."""
        df = get_game_context(2024, 1, 'BAL')
        assert len(df) == 1
        row = df.iloc[0]
        assert row['is_home'] == False

    def test_implied_total_computed(self):
        df = get_game_context(2024, 1, 'KC')
        row = df.iloc[0]
        assert row['implied_total'] is not None
        assert 15 < row['implied_total'] < 40

    def test_rest_days_computed(self):
        df = get_game_context(2024, 1, 'KC')
        row = df.iloc[0]
        assert row['rest_days'] >= 5

    def test_nonexistent_game_returns_empty(self):
        df = get_game_context(2024, 1, 'FAKE')
        assert len(df) == 0


class TestGetOpponentDefenseRank:

    def test_returns_dict(self):
        result = get_opponent_defense_rank('BAL', 'QB', 2024, 10)
        assert isinstance(result, dict)
        assert 'rank' in result
        assert 'avg_pts_allowed' in result
        assert 'games_played' in result

    def test_rank_in_valid_range(self):
        result = get_opponent_defense_rank('BAL', 'QB', 2024, 10)
        assert 1 <= result['rank'] <= 32

    def test_week_1_returns_none(self):
        """No prior data at week 1 — should return None."""
        result = get_opponent_defense_rank('KC', 'QB', 2024, 1)
        assert result is None

    def test_games_played_reasonable(self):
        result = get_opponent_defense_rank('KC', 'RB', 2024, 10)
        assert result is not None
        # By week 10, each team faces multiple RBs per game
        assert result['games_played'] > 10


class TestGetNextgenStats:

    def test_mahomes_passing_single_week(self):
        df = get_nextgen_stats(MAHOMES_ID, 2024, 1, 'passing')
        assert len(df) == 1
        assert 'avg_time_to_throw' in df.columns
        assert 1.5 < df.iloc[0]['avg_time_to_throw'] < 5.0

    def test_mahomes_passing_all_weeks(self):
        df = get_nextgen_stats(MAHOMES_ID, 2024, None, 'passing')
        assert len(df) >= 10  # regular season + playoffs
        assert (df['week'] > 0).all()  # no season aggregates

    def test_rushing_stat_type(self):
        # Barkley rushing
        df = get_nextgen_stats(BARKLEY_ID, 2024, None, 'rushing')
        assert len(df) >= 5
        assert 'efficiency' in df.columns

    def test_receiving_stat_type(self):
        # Use a known receiver — Ja'Marr Chase (00-0036900)
        df = get_nextgen_stats('00-0036900', 2024, None, 'receiving')
        assert len(df) >= 5
        assert 'avg_separation' in df.columns

    def test_invalid_stat_type_raises(self):
        with pytest.raises(ValueError, match="stat_type must be"):
            get_nextgen_stats(MAHOMES_ID, 2024, 1, 'invalid')

    def test_nonexistent_player_returns_empty(self):
        df = get_nextgen_stats('FAKE-ID', 2024, 1, 'passing')
        assert len(df) == 0
