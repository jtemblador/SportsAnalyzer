"""
Tests for the unified NFLDataPipeline — verifies all fetchers are
registered and data directories are populated.
"""

import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.data.pipeline import NFLDataPipeline
from src.nfl.data.fetch_schedules import ScheduleFetcher
from src.nfl.data.fetch_injuries import InjuryFetcher
from src.nfl.data.fetch_snap_counts import SnapCountFetcher
from src.nfl.data.fetch_nextgen_stats import NextGenStatsFetcher
from src.nfl.data.fetch_ff_opportunity import FFOpportunityFetcher
from src.nfl.data.fetch_pfr_advstats import PFRAdvStatsFetcher
from src.nfl.data.fetch_team_stats import TeamStatsFetcher
from src.nfl.data.fetch_depth_charts import DepthChartFetcher


@pytest.fixture
def pipeline():
    return NFLDataPipeline()


class TestFetcherRegistration:
    """Verify all 8 fetchers are registered in the pipeline."""

    def test_schedule_fetcher(self, pipeline):
        assert isinstance(pipeline.schedule_fetcher, ScheduleFetcher)

    def test_injury_fetcher(self, pipeline):
        assert isinstance(pipeline.injury_fetcher, InjuryFetcher)

    def test_snap_count_fetcher(self, pipeline):
        assert isinstance(pipeline.snap_count_fetcher, SnapCountFetcher)

    def test_nextgen_stats_fetcher(self, pipeline):
        assert isinstance(pipeline.nextgen_stats_fetcher, NextGenStatsFetcher)

    def test_ff_opportunity_fetcher(self, pipeline):
        assert isinstance(pipeline.ff_opportunity_fetcher, FFOpportunityFetcher)

    def test_pfr_advstats_fetcher(self, pipeline):
        assert isinstance(pipeline.pfr_advstats_fetcher, PFRAdvStatsFetcher)

    def test_team_stats_fetcher(self, pipeline):
        assert isinstance(pipeline.team_stats_fetcher, TeamStatsFetcher)

    def test_depth_chart_fetcher(self, pipeline):
        assert isinstance(pipeline.depth_chart_fetcher, DepthChartFetcher)


class TestPipelineMethods:
    """Verify fetch_all and fetch_latest exist and are callable."""

    def test_fetch_all_exists(self, pipeline):
        assert hasattr(pipeline, 'fetch_all')
        assert callable(pipeline.fetch_all)

    def test_fetch_latest_exists(self, pipeline):
        assert hasattr(pipeline, 'fetch_latest')
        assert callable(pipeline.fetch_latest)

    def test_player_stats_fetcher_exists(self, pipeline):
        assert hasattr(pipeline, 'player_stats_fetcher')
        assert callable(pipeline.player_stats_fetcher.fetch_all)


class TestDataDirectoriesPopulated:
    """Verify all expected data directories have files on disk."""

    DATA_ROOT = ROOT / "data" / "nfl"

    def test_raw_player_stats_exist(self):
        files = list((self.DATA_ROOT / "raw").glob("*.parquet"))
        assert len(files) >= 100, f"Only {len(files)} raw files"

    def test_schedules_exist(self):
        files = list((self.DATA_ROOT / "schedules").glob("*.parquet"))
        assert len(files) == 8, f"{len(files)} schedule files (expected 8)"

    def test_injuries_exist(self):
        files = list((self.DATA_ROOT / "injuries").glob("*.parquet"))
        assert len(files) == 8, f"{len(files)} injury files (expected 8)"

    def test_snap_counts_exist(self):
        files = list((self.DATA_ROOT / "snap_counts").glob("*.parquet"))
        assert len(files) == 8, f"{len(files)} snap count files (expected 8)"

    def test_nextgen_stats_exist(self):
        files = list((self.DATA_ROOT / "nextgen_stats").glob("*.parquet"))
        assert len(files) == 24, f"{len(files)} NGS files (expected 24)"

    def test_ff_opportunity_exist(self):
        files = list((self.DATA_ROOT / "ff_opportunity").glob("*.parquet"))
        assert len(files) == 8, f"{len(files)} FF opp files (expected 8)"

    def test_pfr_advstats_exist(self):
        files = list((self.DATA_ROOT / "pfr_advstats").glob("*.parquet"))
        assert len(files) == 24, f"{len(files)} PFR files (expected 24)"

    def test_team_stats_exist(self):
        files = list((self.DATA_ROOT / "team_stats").glob("*.parquet"))
        assert len(files) == 8, f"{len(files)} team stat files (expected 8)"

    def test_depth_charts_exist(self):
        files = list((self.DATA_ROOT / "depth_charts").glob("*.parquet"))
        assert len(files) == 7, f"{len(files)} depth chart files (expected 7)"

    def test_total_parquet_files(self):
        """Verify we have the expected total across all datasets."""
        total = 0
        for subdir in ['raw', 'schedules', 'injuries', 'snap_counts',
                       'nextgen_stats', 'ff_opportunity', 'pfr_advstats',
                       'team_stats', 'depth_charts']:
            path = self.DATA_ROOT / subdir
            if path.exists():
                total += len(list(path.glob("*.parquet")))
        # 144 raw + 8 schedules + 8 injuries + 8 snap + 24 ngs + 8 ff + 24 pfr + 8 team + 7 depth = 239+
        assert total >= 230, f"Only {total} total Parquet files"
