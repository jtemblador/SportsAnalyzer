"""
Tests for the PlayerStatsFetcher — verifies per-season player stats
are fetched correctly.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_player_stats import PlayerStatsFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return PlayerStatsFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return PlayerStatsFetcher()


class TestPlayerStatsFetcher:

    def test_instantiation(self, fetcher):
        assert fetcher is not None
        assert fetcher.data_dir.exists()

    def test_fetch_season_creates_file(self, fetcher):
        result = fetcher.fetch_season(2024)
        assert result is not None
        assert fetcher._file_path(2024).exists()

    def test_skip_existing(self, fetcher):
        fetcher.fetch_season(2024)
        result = fetcher.fetch_season(2024)
        assert result is None

    def test_force_refetch(self, fetcher):
        fetcher.fetch_season(2024)
        result = fetcher.fetch_season(2024, force=True)
        assert result is not None

    def test_fetch_all_creates_multiple_files(self, fetcher):
        fetcher.fetch_all(start_season=2023, end_season=2024)
        assert fetcher._file_path(2023).exists()
        assert fetcher._file_path(2024).exists()


class TestPlayerStatsDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        assert len(df.columns) >= 114

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'player_id', 'player_name', 'position', 'team', 'week', 'season',
            'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
            'receptions', 'receiving_yards', 'receiving_tds', 'targets',
            'fantasy_points', 'fantasy_points_ppr',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_includes_playoff_weeks(self, production_fetcher):
        """New per-season files should include playoff weeks (19+)."""
        df = production_fetcher.load_season(2024)
        assert df['week'].max() >= 19, f"Max week {df['week'].max()}, expected 19+"

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert 15000 <= len(df) <= 22000, f"{season}: {len(df)} rows"

    def test_player_id_mostly_populated(self, production_fetcher):
        """player_id has a small number of nulls (~0.1%) from nflreadpy — acceptable."""
        df = production_fetcher.load_season(2024)
        null_pct = df['player_id'].isna().mean()
        assert null_pct < 0.01, f"player_id {null_pct*100:.1f}% null (expected <1%)"

    def test_all_seasons_fetched(self, production_fetcher):
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing player stats for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:

    def test_pipeline_has_player_stats_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'player_stats_fetcher')
        assert isinstance(pipeline.player_stats_fetcher, PlayerStatsFetcher)
