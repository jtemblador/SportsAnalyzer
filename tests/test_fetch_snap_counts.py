"""
Tests for the SnapCountFetcher — verifies weekly snap count data
(offense/defense/ST snaps and percentages) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_snap_counts import SnapCountFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return SnapCountFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return SnapCountFetcher()


class TestSnapCountFetcher:

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


class TestSnapCountDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        assert len(df.columns) == 16

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'game_id', 'season', 'week', 'player', 'pfr_player_id',
            'position', 'team', 'opponent',
            'offense_snaps', 'offense_pct',
            'defense_snaps', 'defense_pct',
            'st_snaps', 'st_pct',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2019, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert df is not None
            assert 20000 <= len(df) <= 30000, f"{season}: {len(df)} rows"

    def test_zero_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        total_nulls = df.isna().sum().sum()
        assert total_nulls == 0, f"Found {total_nulls} nulls"

    def test_offense_pct_range(self, production_fetcher):
        """offense_pct should be 0.0-1.0, not 0-100."""
        df = production_fetcher.load_season(2024)
        assert df['offense_pct'].min() >= 0.0
        assert df['offense_pct'].max() <= 1.0

    def test_spot_check_derrick_henry_2024(self, production_fetcher):
        """Derrick Henry 2024 Week 1: BAL RB, ~46% offense snaps."""
        df = production_fetcher.load_season(2024)
        henry = df[(df['player'] == 'Derrick Henry') & (df['week'] == 1)]
        assert len(henry) == 1
        row = henry.iloc[0]
        assert row['team'] == 'BAL'
        assert row['position'] == 'RB'
        assert 0.3 <= row['offense_pct'] <= 0.7

    def test_all_seasons_fetched(self, production_fetcher):
        """Verify we have snap count data for every season 2018-2025."""
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing snap count data for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:
    """Verify SnapCountFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_snap_count_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'snap_count_fetcher')
        assert isinstance(pipeline.snap_count_fetcher, SnapCountFetcher)
