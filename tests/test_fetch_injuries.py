"""
Tests for the InjuryFetcher — verifies weekly injury report data
(game-day status, practice participation, injury type) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_injuries import InjuryFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return InjuryFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return InjuryFetcher()


class TestInjuryFetcher:

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


class TestInjuryDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        assert len(df.columns) == 16

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'season', 'week', 'gsis_id', 'full_name', 'position', 'team',
            'report_status', 'report_primary_injury',
            'practice_status', 'practice_primary_injury',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2019, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert df is not None
            assert 4000 <= len(df) <= 7000, f"{season}: {len(df)} rows"

    def test_gsis_id_no_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df['gsis_id'].isna().sum() == 0

    def test_report_status_valid_values(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        valid = {'Out', 'Doubtful', 'Questionable', 'Note', None}
        actual = set(df['report_status'].dropna().unique())
        unexpected = actual - valid
        assert len(unexpected) == 0, f"Unexpected report_status values: {unexpected}"

    def test_practice_status_mostly_populated(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        # practice_status should be ~99%+ populated
        pct = df['practice_status'].notna().mean()
        assert pct > 0.95, f"practice_status only {pct*100:.1f}% populated"

    def test_spot_check_mahomes_2024_week_18(self, production_fetcher):
        """Mahomes was Doubtful with ankle injury in Week 18, 2024."""
        df = production_fetcher.load_season(2024)
        mahomes = df[(df['full_name'] == 'Patrick Mahomes') & (df['week'] == 18)]
        assert len(mahomes) == 1
        row = mahomes.iloc[0]
        assert row['report_status'] == 'Doubtful'
        assert row['report_primary_injury'] == 'Ankle'

    def test_all_seasons_fetched(self, production_fetcher):
        """Verify we have injury data for every season 2018-2025."""
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing injury data for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:
    """Verify InjuryFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_injury_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'injury_fetcher')
        assert isinstance(pipeline.injury_fetcher, InjuryFetcher)
