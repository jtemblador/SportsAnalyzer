"""
Tests for the FFOpportunityFetcher — verifies expected fantasy points data
(actual vs expected, team shares) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_ff_opportunity import FFOpportunityFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return FFOpportunityFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return FFOpportunityFetcher()


class TestFFOpportunityFetcher:

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


class TestFFOpportunityDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        assert len(df.columns) == 159

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'player_id', 'full_name', 'position', 'week',
            'total_fantasy_points', 'total_fantasy_points_exp',
            'total_fantasy_points_diff',
            'pass_fantasy_points_exp', 'rec_fantasy_points_exp',
            'rush_fantasy_points_exp',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2019, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert df is not None
            assert 5000 <= len(df) <= 7000, f"{season}: {len(df)} rows"

    def test_fantasy_points_columns_no_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        for col in ['total_fantasy_points', 'total_fantasy_points_exp', 'total_fantasy_points_diff']:
            assert df[col].isna().sum() == 0, f"{col} has nulls"

    def test_spot_check_mahomes_2024_week_1(self, production_fetcher):
        """Mahomes 2024 Week 1: actual ~15.14, expected ~13.14, diff ~2.0."""
        df = production_fetcher.load_season(2024)
        mahomes = df[(df['full_name'] == 'Patrick Mahomes') & (df['week'] == 1)]
        assert len(mahomes) == 1
        row = mahomes.iloc[0]
        assert 10 <= row['total_fantasy_points'] <= 25
        assert 8 <= row['total_fantasy_points_exp'] <= 25
        assert abs(row['total_fantasy_points_diff'] - (row['total_fantasy_points'] - row['total_fantasy_points_exp'])) < 0.01

    def test_diff_equals_actual_minus_expected(self, production_fetcher):
        """Verify diff column = actual - expected for all rows."""
        df = production_fetcher.load_season(2024)
        calculated = df['total_fantasy_points'] - df['total_fantasy_points_exp']
        max_error = (df['total_fantasy_points_diff'] - calculated).abs().max()
        assert max_error < 0.01, f"Diff column doesn't match actual - expected, max error: {max_error}"

    def test_all_seasons_fetched(self, production_fetcher):
        """Verify we have FF opportunity data for every season 2018-2025."""
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing FF opportunity data for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:
    """Verify FFOpportunityFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_ff_opportunity_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'ff_opportunity_fetcher')
        assert isinstance(pipeline.ff_opportunity_fetcher, FFOpportunityFetcher)
