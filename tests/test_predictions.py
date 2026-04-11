"""
Tests for the predictions and model_versions tables.
Verifies data was loaded correctly and cross-version accuracy queries work.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.nfl.db.connection import get_connection, get_engine


@pytest.fixture
def conn():
    c = get_connection()
    yield c
    c.close()


@pytest.fixture
def engine():
    return get_engine()


class TestModelVersions:

    def test_four_versions_loaded(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM model_versions")
        assert cur.fetchone()[0] == 4
        cur.close()

    def test_version_names(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT version FROM model_versions ORDER BY version")
        versions = [row[0] for row in cur.fetchall()]
        cur.close()
        assert 'v1_baseline_mae5.14' in versions
        assert 'v2_variance_trends_mae4.66' in versions
        assert 'v3_epa_efficiency' in versions
        assert 'v4_position_specific' in versions

    def test_mae_values(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT version, mae FROM model_versions ORDER BY mae")
        rows = cur.fetchall()
        cur.close()
        # V4 should have lowest MAE
        assert rows[0][0] == 'v4_position_specific'
        assert rows[0][1] == pytest.approx(4.26)
        # V1 should have highest MAE
        assert rows[-1][0] == 'v1_baseline_mae5.14'
        assert rows[-1][1] == pytest.approx(5.14)

    def test_all_fields_populated(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT * FROM model_versions")
        for row in cur.fetchall():
            assert row[0] is not None  # version
            assert row[1] is not None  # description
            assert row[2] is not None  # mae
            assert row[3] is not None  # prediction_weeks
            assert row[4] is not None  # prediction_season
        cur.close()


class TestPredictionsLoaded:

    def test_total_rows(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.close()
        assert total == 65921

    def test_rows_per_version(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT version, COUNT(*) FROM predictions GROUP BY version ORDER BY version")
        rows = {row[0]: row[1] for row in cur.fetchall()}
        cur.close()
        assert rows['v1_baseline_mae5.14'] == 12882
        assert rows['v4_position_specific'] == 17759

    def test_all_positions_present(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT position FROM predictions ORDER BY position")
        positions = [row[0] for row in cur.fetchall()]
        cur.close()
        assert set(positions) == {'K', 'QB', 'RB', 'TE', 'WR'}

    def test_all_model_types_present(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT model_type FROM predictions ORDER BY model_type")
        types = [row[0] for row in cur.fetchall()]
        cur.close()
        assert 'evob' in types
        assert 'pob' in types

    def test_predictions_reference_valid_versions(self, conn):
        """Every prediction's version should exist in model_versions."""
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM predictions p
            LEFT JOIN model_versions mv ON p.version = mv.version
            WHERE mv.version IS NULL
        """)
        orphans = cur.fetchone()[0]
        cur.close()
        assert orphans == 0


class TestActualsBackfilled:

    def test_actuals_populated(self, conn):
        """Most predictions should have actuals backfilled."""
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE actual_value IS NOT NULL AND predicted_value IS NOT NULL
        """)
        with_actuals = cur.fetchone()[0]
        cur.close()
        assert with_actuals > 50000

    def test_error_computed(self, conn):
        """Error should be populated wherever actual_value is."""
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE actual_value IS NOT NULL AND error IS NULL
              AND predicted_value IS NOT NULL
        """)
        missing_error = cur.fetchone()[0]
        cur.close()
        assert missing_error == 0

    def test_error_is_absolute(self, conn):
        """Error should always be >= 0."""
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions WHERE error < 0")
        negative = cur.fetchone()[0]
        cur.close()
        assert negative == 0

    def test_mahomes_fantasy_points_actual(self, engine):
        """Spot-check: Mahomes V4 week 1 actual should match weekly_stats."""
        pred = pd.read_sql(
            "SELECT actual_value FROM predictions "
            "WHERE version = 'v4_position_specific' "
            "AND player_name = 'P.Mahomes' AND week = 1 AND season = 2025 "
            "AND stat = 'fantasy_points_ppr' AND model_type = 'evob'",
            engine
        )
        actual_ws = pd.read_sql(
            "SELECT fantasy_points_ppr FROM weekly_stats "
            "WHERE player_id = '00-0033873' AND season = 2025 AND week = 1",
            engine
        )
        if not pred.empty and not actual_ws.empty:
            assert pred.iloc[0]['actual_value'] == pytest.approx(
                actual_ws.iloc[0]['fantasy_points_ppr']
            )


class TestCrossVersionAccuracy:

    def test_v4_best_mae_fantasy_points(self, engine):
        """V4 should have the lowest MAE for fantasy_points_ppr."""
        df = pd.read_sql("""
            SELECT version, AVG(error) as mae
            FROM predictions
            WHERE stat = 'fantasy_points_ppr'
              AND actual_value IS NOT NULL
              AND predicted_value IS NOT NULL
            GROUP BY version
            ORDER BY mae
        """, engine)
        assert len(df) == 4
        assert df.iloc[0]['version'] == 'v4_position_specific'

    def test_v1_worst_mae_fantasy_points(self, engine):
        """V1 should have the highest MAE for fantasy_points_ppr."""
        df = pd.read_sql("""
            SELECT version, AVG(error) as mae
            FROM predictions
            WHERE stat = 'fantasy_points_ppr'
              AND actual_value IS NOT NULL
              AND predicted_value IS NOT NULL
            GROUP BY version
            ORDER BY mae DESC
        """, engine)
        assert df.iloc[0]['version'] == 'v1_baseline_mae5.14'

    def test_te_lowest_mae_v4(self, engine):
        """TE should have the lowest MAE in V4 (most predictable position)."""
        df = pd.read_sql("""
            SELECT position, AVG(error) as mae
            FROM predictions
            WHERE version = 'v4_position_specific'
              AND stat = 'fantasy_points_ppr'
              AND actual_value IS NOT NULL
              AND predicted_value IS NOT NULL
            GROUP BY position
            ORDER BY mae
        """, engine)
        assert df.iloc[0]['position'] == 'TE'

    def test_mae_values_reasonable(self, engine):
        """All MAE values should be between 1 and 15 for fantasy_points_ppr."""
        df = pd.read_sql("""
            SELECT version, AVG(error) as mae
            FROM predictions
            WHERE stat = 'fantasy_points_ppr'
              AND actual_value IS NOT NULL
              AND predicted_value IS NOT NULL
            GROUP BY version
        """, engine)
        for _, row in df.iterrows():
            assert 1 < row['mae'] < 15, f"{row['version']} MAE {row['mae']} out of range"
