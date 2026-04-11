"""
Tests for the DepthChartFetcher — verifies weekly depth chart data
(starter/backup status, formation, position) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_depth_charts import DepthChartFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return DepthChartFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return DepthChartFetcher()


class TestDepthChartFetcher:

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

    def test_2025_skipped(self, fetcher):
        """2025 has incompatible schema — should return None."""
        result = fetcher.fetch_season(2025)
        assert result is None
        assert not fetcher._file_path(2025).exists()


class TestDepthChartDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        assert len(df.columns) == 15

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'season', 'club_code', 'week', 'depth_team',
            'formation', 'gsis_id', 'position', 'depth_position',
            'full_name',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_gsis_id_no_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df['gsis_id'].isna().sum() == 0

    def test_depth_team_valid_values(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        valid = {'1', '2', '3'}
        actual = set(df['depth_team'].unique())
        assert actual == valid, f"Unexpected depth_team values: {actual - valid}"

    def test_formation_values(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        expected = {'Offense', 'Defense', 'Special Teams'}
        actual = set(df['formation'].unique())
        assert actual == expected

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert 30000 <= len(df) <= 40000, f"{season}: {len(df)} rows"

    def test_seasons_2018_to_2024_fetched(self, production_fetcher):
        for season in range(2018, 2025):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing depth chart data for {season}"
            assert len(df) > 0

    def test_2025_not_on_disk(self, production_fetcher):
        result = production_fetcher.load_season(2025)
        assert result is None

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:

    def test_pipeline_has_depth_chart_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'depth_chart_fetcher')
        assert isinstance(pipeline.depth_chart_fetcher, DepthChartFetcher)
