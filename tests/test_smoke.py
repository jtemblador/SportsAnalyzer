"""
Smoke tests to verify the NFL prediction pipeline is functional.
Run before and after reorganization to ensure nothing broke.
"""

import sys
import pytest
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestImports:
    """Verify all core modules can be imported."""

    def test_import_pipeline(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        assert NFLDataPipeline is not None

    def test_import_db_connection(self):
        from src.nfl.db.connection import get_connection, get_engine
        assert get_connection is not None
        assert get_engine is not None

    def test_import_db_queries(self):
        from src.nfl.db.queries import get_player_history, get_game_context
        assert get_player_history is not None
        assert get_game_context is not None


class TestDataFiles:
    """Verify data files exist and are readable."""

    def test_player_stats_exist(self):
        stats_dir = ROOT / "data" / "nfl" / "player_stats"
        parquet_files = list(stats_dir.glob("*.parquet"))
        assert len(parquet_files) >= 8, "Expected at least 8 per-season player stats files"

    def test_feature_files_exist(self):
        features_dir = ROOT / "data" / "nfl" / "features" / "v4_position_specific"
        parquet_files = list(features_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No V4 feature files found"

    def test_model_files_exist(self):
        models_dir = ROOT / "data" / "nfl" / "models" / "v4_position_specific"
        joblib_files = list(models_dir.glob("*.joblib"))
        assert len(joblib_files) > 0, "No V4 model files found"

    def test_prediction_files_exist(self):
        pred_dir = ROOT / "data" / "nfl" / "predictions" / "v4_position_specific"
        parquet_files = list(pred_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No V4 prediction files found"

    def test_player_stats_readable(self):
        import pandas as pd
        stats_dir = ROOT / "data" / "nfl" / "player_stats"
        first_file = sorted(stats_dir.glob("*.parquet"))[0]
        df = pd.read_parquet(first_file)
        assert len(df) > 0
        assert "player_name" in df.columns

    def test_prediction_data_readable(self):
        import pandas as pd
        pred_dir = ROOT / "data" / "nfl" / "predictions" / "v4_position_specific"
        first_file = sorted(pred_dir.glob("*.parquet"))[0]
        df = pd.read_parquet(first_file)
        assert len(df) > 0


class TestPipelineInstantiation:
    """Verify core classes can be instantiated."""

    def test_pipeline_init(self):
        from src.nfl.data.pipeline import NFLDataPipeline
        pipeline = NFLDataPipeline(base_data_dir=str(ROOT / "data"))
        assert pipeline is not None
        assert pipeline.player_stats_fetcher is not None

    def test_model_loading(self):
        import joblib
        models_dir = ROOT / "data" / "nfl" / "models" / "v4_position_specific"
        first_model = sorted(models_dir.glob("*.joblib"))[0]
        model = joblib.load(first_model)
        assert model is not None
