"""
Tests for the ScheduleFetcher — verifies schedule data
(Vegas lines, weather, game context) is fetched and stored correctly.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_schedules import ScheduleFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return ScheduleFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return ScheduleFetcher()


class TestScheduleFetcher:

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
        assert result is None  # skipped

    def test_force_refetch(self, fetcher):
        fetcher.fetch_season(2024)
        result = fetcher.fetch_season(2024, force=True)
        assert result is not None  # re-fetched

    def test_fetch_all_creates_multiple_files(self, fetcher):
        fetcher.fetch_all(start_season=2023, end_season=2024)
        assert fetcher._file_path(2023).exists()
        assert fetcher._file_path(2024).exists()


class TestScheduleDataQuality:
    """Tests against real fetched data on disk."""

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df is not None
        # 46 original + 2 derived (home_implied_total, away_implied_total)
        assert len(df.columns) == 48

    def test_expected_columns_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        required = [
            'game_id', 'season', 'week', 'home_team', 'away_team',
            'spread_line', 'total_line', 'home_moneyline', 'away_moneyline',
            'temp', 'wind', 'roof', 'surface',
            'home_rest', 'away_rest', 'div_game',
            'home_score', 'away_score',
            'home_implied_total', 'away_implied_total',
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_reasonable(self, production_fetcher):
        for season in [2018, 2019, 2020, 2024]:
            df = production_fetcher.load_season(season)
            assert df is not None
            assert 250 <= len(df) <= 300, f"{season}: {len(df)} games"

    def test_spread_line_no_nulls(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df['spread_line'].isna().sum() == 0

    def test_implied_totals_correct(self, production_fetcher):
        """Spot-check: 2024 Week 1 BAL@KC — spread 3.0, total 46.0"""
        df = production_fetcher.load_season(2024)
        game = df[(df['week'] == 1) & (df['home_team'] == 'KC') & (df['away_team'] == 'BAL')]
        assert len(game) == 1
        row = game.iloc[0]
        assert row['spread_line'] == 3.0
        assert row['total_line'] == 46.0
        assert row['home_implied_total'] == 24.5
        assert row['away_implied_total'] == 21.5

    def test_weather_fields_present(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        # Roof should have no nulls
        assert df['roof'].isna().sum() == 0
        # Surface should have no nulls
        assert df['surface'].isna().sum() == 0
        # Temp/wind can be null for dome games — but should have some non-null values
        assert df['temp'].notna().sum() > 100

    def test_rest_days_valid(self, production_fetcher):
        df = production_fetcher.load_season(2024)
        assert df['home_rest'].isna().sum() == 0
        assert df['away_rest'].isna().sum() == 0
        # Rest days should be between 3 and 21 (bye weeks can extend rest)
        assert df['home_rest'].min() >= 3
        assert df['away_rest'].min() >= 3

    def test_2018_2019_are_17_weeks(self, production_fetcher):
        """2018-2019 were 16-game seasons (weeks 1-17 including playoffs)."""
        for season in [2018, 2019]:
            df = production_fetcher.load_season(season)
            assert df is not None
            # Should have regular season + postseason
            assert df['week'].max() >= 17

    def test_implied_totals_math_all_games(self, production_fetcher):
        """Verify implied totals add up to total_line for every game."""
        df = production_fetcher.load_season(2024)
        # home_implied + away_implied should always equal total_line
        reconstructed = df['home_implied_total'] + df['away_implied_total']
        diff = (reconstructed - df['total_line']).abs()
        assert diff.max() < 0.01, f"Implied totals don't sum to total_line, max diff: {diff.max()}"

    def test_all_seasons_fetched(self, production_fetcher):
        """Verify we have schedule data for every season 2018-2025."""
        for season in range(2018, 2026):
            df = production_fetcher.load_season(season)
            assert df is not None, f"Missing schedule data for {season}"
            assert len(df) > 0

    def test_load_nonexistent_season_returns_none(self, production_fetcher):
        result = production_fetcher.load_season(1999)
        assert result is None


class TestPipelineIntegration:
    """Verify ScheduleFetcher is wired into NFLDataPipeline."""

    def test_pipeline_has_schedule_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'schedule_fetcher')
        assert isinstance(pipeline.schedule_fetcher, ScheduleFetcher)

    def test_pipeline_fetch_all_exists(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'fetch_all')
        assert callable(pipeline.fetch_all)
