"""
Tests for the PFRAdvStatsFetcher — verifies PFR Advanced Stats data
(pass pressure, rush contact, rec drops) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_pfr_advstats import PFRAdvStatsFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return PFRAdvStatsFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return PFRAdvStatsFetcher()


class TestPFRAdvStatsFetcher:

    def test_instantiation(self, fetcher):
        assert fetcher is not None
        assert fetcher.data_dir.exists()

    def test_fetch_season_creates_file(self, fetcher):
        result = fetcher.fetch_season(2024, 'pass')
        assert result is not None
        assert fetcher._file_path(2024, 'pass').exists()

    def test_skip_existing(self, fetcher):
        fetcher.fetch_season(2024, 'pass')
        result = fetcher.fetch_season(2024, 'pass')
        assert result is None

    def test_force_refetch(self, fetcher):
        fetcher.fetch_season(2024, 'pass')
        result = fetcher.fetch_season(2024, 'pass', force=True)
        assert result is not None

    def test_fetch_all_creates_all_files(self, fetcher):
        fetcher.fetch_all(start_season=2024, end_season=2024)
        for stat_type in ['pass', 'rush', 'rec']:
            assert fetcher._file_path(2024, stat_type).exists()


class TestPFRDataQuality:
    """Tests against real fetched data on disk."""

    def test_pass_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'pass')
        assert df is not None
        assert len(df.columns) == 24

    def test_rush_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rush')
        assert len(df.columns) == 16

    def test_rec_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rec')
        assert len(df.columns) == 17

    def test_pass_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'pass')
        required = [
            'season', 'week', 'pfr_player_name', 'pfr_player_id',
            'team', 'opponent',
            'passing_drops', 'passing_bad_throws',
            'times_pressured', 'times_pressured_pct',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_rush_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rush')
        required = [
            'season', 'week', 'pfr_player_name', 'pfr_player_id',
            'rushing_yards_before_contact', 'rushing_yards_after_contact',
            'rushing_broken_tackles',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_rec_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rec')
        required = [
            'season', 'week', 'pfr_player_name', 'pfr_player_id',
            'receiving_drop', 'receiving_drop_pct',
            'receiving_broken_tackles', 'receiving_rat',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_pfr_player_id_no_nulls(self, production_fetcher):
        for stat_type in ['pass', 'rush', 'rec']:
            df = production_fetcher.load_season(2024, stat_type)
            assert df['pfr_player_id'].isna().sum() == 0, f"{stat_type}: has null pfr_player_ids"

    def test_all_seasons_and_types_fetched(self, production_fetcher):
        """Verify we have PFR data for every season 2018-2025 x 3 types."""
        for season in range(2018, 2026):
            for stat_type in ['pass', 'rush', 'rec']:
                df = production_fetcher.load_season(season, stat_type)
                assert df is not None, f"Missing PFR {stat_type} for {season}"
                assert len(df) > 0

    def test_load_nonexistent_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999, 'pass')
        assert result is None


class TestPipelineIntegration:
    """Verify PFRAdvStatsFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_pfr_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'pfr_advstats_fetcher')
        assert isinstance(pipeline.pfr_advstats_fetcher, PFRAdvStatsFetcher)
