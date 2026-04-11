"""
Tests for the TeamStatsFetcher — verifies team-level weekly stats
(offensive/defensive EPA, yards, TDs, turnovers) are fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_team_stats import TeamStatsFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return TeamStatsFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return TeamStatsFetcher()


class TestTeamStatsFetcher:

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


class TestTeamStatsDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        # 102-103 columns depending on season
        assert 100 <= len(df.columns) <= 105

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'season', 'week', 'team',
            'passing_yards', 'passing_tds', 'passing_epa',
            'rushing_yards', 'rushing_tds', 'rushing_epa',
            'receiving_yards', 'receiving_tds', 'receiving_epa',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_32_teams_per_season(self, production_fetcher):
        for season in [2018, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert df['team'].nunique() == 32, f"{season}: {df['team'].nunique()} teams"

    def test_team_no_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df['team'].isna().sum() == 0

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2019, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert 500 <= len(df) <= 600, f"{season}: {len(df)} rows"

    def test_all_seasons_fetched(self, production_fetcher):
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing team stats for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:

    def test_pipeline_has_team_stats_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'team_stats_fetcher')
        assert isinstance(pipeline.team_stats_fetcher, TeamStatsFetcher)
