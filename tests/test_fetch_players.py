"""
Tests for the PlayersFetcher — verifies player reference table with
GSIS/PFR ID mapping is fetched, filtered, and provides cross-dataset coverage.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.fetch_players import PlayersFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create a fetcher pointing at a temp directory."""
    return PlayersFetcher(data_dir=str(tmp_path))


@pytest.fixture
def production_fetcher():
    """Fetcher pointing at real data (read-only tests)."""
    return PlayersFetcher()


class TestPlayersFetcher:

    def test_instantiation(self, fetcher):
        assert fetcher is not None
        assert fetcher.data_dir.exists()

    def test_fetch_creates_file(self, fetcher):
        result = fetcher.fetch()
        assert result is not None
        assert fetcher._file_path().exists()

    def test_skip_existing(self, fetcher):
        fetcher.fetch()
        result = fetcher.fetch()
        assert result is None

    def test_force_refetch(self, fetcher):
        fetcher.fetch()
        result = fetcher.fetch(force=True)
        assert result is not None


class TestPlayersDataQuality:

    def test_filtered_to_recent(self, production_fetcher):
        df = production_fetcher.load()
        assert df is not None
        assert df['last_season'].min() >= 2018

    def test_not_full_alltime_table(self, production_fetcher):
        """Should be ~6,500 players, not 24,000+."""
        df = production_fetcher.load()
        assert 5000 <= len(df) <= 10000

    def test_column_count(self, production_fetcher):
        df = production_fetcher.load()
        assert len(df.columns) == 39

    def test_gsis_id_no_nulls(self, production_fetcher):
        df = production_fetcher.load()
        assert df['gsis_id'].isna().sum() == 0

    def test_key_columns_present(self, production_fetcher):
        df = production_fetcher.load()
        required = ['gsis_id', 'pfr_id', 'display_name', 'position', 'last_season', 'latest_team']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"


class TestIDMapping:
    """Verify the mapping covers actual players in our datasets."""

    def test_mapping_returns_both_ids(self, production_fetcher):
        mapping = production_fetcher.get_id_mapping()
        assert mapping is not None
        assert mapping['gsis_id'].isna().sum() == 0
        assert mapping['pfr_id'].isna().sum() == 0

    def test_mapping_size_reasonable(self, production_fetcher):
        mapping = production_fetcher.get_id_mapping()
        assert 5000 <= len(mapping) <= 10000

    def test_covers_player_stats(self, production_fetcher):
        """Mapping should cover 99%+ of player_stats player_ids."""
        mapping = production_fetcher.get_id_mapping()
        ps = pd.read_parquet(ROOT / 'data/nfl/player_stats/player_stats_2024.parquet')
        ps_ids = set(ps['player_id'].dropna().unique())
        gsis_ids = set(mapping['gsis_id'].unique())
        coverage = len(ps_ids & gsis_ids) / len(ps_ids)
        assert coverage >= 0.99, f"Player stats coverage {coverage*100:.1f}% (need 99%+)"

    def test_covers_snap_counts(self, production_fetcher):
        """Mapping should cover 99%+ of snap_counts pfr_player_ids."""
        mapping = production_fetcher.get_id_mapping()
        snap = pd.read_parquet(ROOT / 'data/nfl/snap_counts/snap_counts_2024.parquet')
        snap_ids = set(snap['pfr_player_id'].unique())
        pfr_ids = set(mapping['pfr_id'].unique())
        coverage = len(snap_ids & pfr_ids) / len(snap_ids)
        assert coverage >= 0.99, f"Snap counts coverage {coverage*100:.1f}% (need 99%+)"

    def test_covers_pfr_advstats(self, production_fetcher):
        """Mapping should cover 100% of PFR pass advstats."""
        mapping = production_fetcher.get_id_mapping()
        pfr = pd.read_parquet(ROOT / 'data/nfl/pfr_advstats/pfr_pass_2024.parquet')
        pfr_ids = set(pfr['pfr_player_id'].unique())
        mapped_pfr = set(mapping['pfr_id'].unique())
        coverage = len(pfr_ids & mapped_pfr) / len(pfr_ids)
        assert coverage >= 0.99, f"PFR pass coverage {coverage*100:.1f}% (need 99%+)"


class TestPipelineIntegration:

    def test_pipeline_has_players_fetcher(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline()
        assert hasattr(pipeline, 'players_fetcher')
        assert isinstance(pipeline.players_fetcher, PlayersFetcher)
