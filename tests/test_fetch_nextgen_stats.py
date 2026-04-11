"""
Tests for the NextGenStatsFetcher — verifies Next Gen Stats data
(passing, rushing, receiving) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_nextgen_stats import NextGenStatsFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return NextGenStatsFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return NextGenStatsFetcher()


class TestNextGenStatsFetcher:

    def test_instantiation(self, fetcher):
        assert fetcher is not None
        assert fetcher.data_dir.exists()

    def test_fetch_season_creates_file(self, fetcher):
        result = fetcher.fetch_season(2024, 'passing')
        assert result is not None
        assert fetcher._file_path(2024, 'passing').exists()

    def test_skip_existing(self, fetcher):
        fetcher.fetch_season(2024, 'passing')
        result = fetcher.fetch_season(2024, 'passing')
        assert result is None

    def test_force_refetch(self, fetcher):
        fetcher.fetch_season(2024, 'passing')
        result = fetcher.fetch_season(2024, 'passing', force=True)
        assert result is not None

    def test_fetch_all_creates_all_files(self, fetcher):
        fetcher.fetch_all(start_season=2024, end_season=2024)
        for stat_type in ['passing', 'rushing', 'receiving']:
            assert fetcher._file_path(2024, stat_type).exists()


class TestNGSDataQuality:
    """Tests against real fetched data on disk."""

    def test_passing_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'passing')
        assert df is not None
        assert len(df.columns) == 29

    def test_rushing_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rushing')
        assert len(df.columns) == 22

    def test_receiving_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'receiving')
        assert len(df.columns) == 23

    def test_passing_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'passing')
        required = [
            'season', 'week', 'player_display_name', 'player_gsis_id',
            'avg_time_to_throw', 'avg_completed_air_yards',
            'aggressiveness', 'completion_percentage_above_expectation',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_rushing_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'rushing')
        required = [
            'season', 'week', 'player_display_name', 'player_gsis_id',
            'efficiency', 'rush_yards_over_expected',
            'rush_yards_over_expected_per_att',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_receiving_key_columns(self, production_fetcher):
        df = production_fetcher.load_season(2024, 'receiving')
        required = [
            'season', 'week', 'player_display_name', 'player_gsis_id',
            'avg_cushion', 'avg_separation',
            'avg_yac_above_expectation', 'catch_percentage',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_gsis_id_no_nulls(self, production_fetcher):
        for stat_type in ['passing', 'rushing', 'receiving']:
            df = production_fetcher.load_season(2024, stat_type)
            assert df['player_gsis_id'].isna().sum() == 0, f"{stat_type}: has null gsis_ids"

    def test_week_0_is_season_aggregate(self, production_fetcher):
        """Week 0 should have high attempt counts (full season totals)."""
        df = production_fetcher.load_season(2024, 'passing')
        w0 = df[df['week'] == 0]
        assert len(w0) > 0, "No week 0 data"
        # Season aggregate should have >100 attempts for starters
        assert w0['attempts'].max() > 100

    def test_spot_check_mahomes_time_to_throw(self, production_fetcher):
        """Mahomes avg_time_to_throw should be ~2.5-3.5s."""
        df = production_fetcher.load_season(2024, 'passing')
        mahomes = df[(df['player_display_name'] == 'Patrick Mahomes') & (df['week'] > 0)]
        assert len(mahomes) > 0
        avg_ttt = mahomes['avg_time_to_throw'].mean()
        assert 2.0 <= avg_ttt <= 4.0, f"Mahomes time_to_throw {avg_ttt:.2f}s out of range"

    def test_spot_check_top_ryoe_positive(self, production_fetcher):
        """Top RBs should have positive rush_yards_over_expected in season aggregate."""
        df = production_fetcher.load_season(2024, 'rushing')
        season_agg = df[df['week'] == 0]
        top = season_agg.nlargest(3, 'rush_yards_over_expected')
        assert top['rush_yards_over_expected'].min() > 0

    def test_all_seasons_and_types_fetched(self, production_fetcher):
        """Verify we have NGS data for every season 2018-2025 x 3 types."""
        for season in range(2018, 2026):
            for stat_type in ['passing', 'rushing', 'receiving']:
                df = production_fetcher.load_season(season, stat_type)
                assert df is not None, f"Missing NGS {stat_type} for {season}"
                assert len(df) > 0

    def test_load_nonexistent_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999, 'passing')
        assert result is None


class TestPipelineIntegration:
    """Verify NextGenStatsFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_ngs_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'nextgen_stats_fetcher')
        assert isinstance(pipeline.nextgen_stats_fetcher, NextGenStatsFetcher)
