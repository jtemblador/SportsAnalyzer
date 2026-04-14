"""
Tests for the PostgreSQL database — verifies schema, data loading,
connection layer, and cross-dataset joins.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import psycopg2
import pandas as pd
from src.nfl.db.config import DB_CONFIG
from src.nfl.db.connection import get_connection, get_engine

# Expected row counts after bulk load (from load_all.py output)
EXPECTED_ROW_COUNTS = {
    'players': 6543,
    'games': 2227,
    'weekly_stats': 147050,
    'injuries': 45337,
    'depth_charts': 258942,
    'snap_counts': 205354,
    'pfr_pass_advstats': 5424,
    'pfr_rush_advstats': 18461,
    'pfr_rec_advstats': 35724,
    'ngs_passing': 4785,
    'ngs_rushing': 4885,
    'ngs_receiving': 11708,
    'ff_opportunity': 47282,
    'team_stats': 4454,
}


@pytest.fixture
def conn():
    """Create a database connection for testing."""
    connection = psycopg2.connect(**DB_CONFIG)
    yield connection
    connection.close()


class TestDatabaseConnection:

    def test_can_connect(self, conn):
        assert conn is not None
        assert conn.closed == 0

    def test_correct_database(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT current_database()")
        db = cur.fetchone()[0]
        assert db == 'nfl_predictions'
        cur.close()


class TestTablesExist:

    def test_all_tables_exist(self, conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        cur.close()
        # 14 data tables + 3 prediction/eval tables (predictions, model_versions,
        # model_eval_metrics added in Task 3.2)
        assert len(tables) == 17, f"Expected 17 tables, found {len(tables)}: {tables}"

    def test_each_table_exists(self, conn):
        cur = conn.cursor()
        for table_name in EXPECTED_ROW_COUNTS:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (table_name,))
            exists = cur.fetchone()[0]
            assert exists, f"Table {table_name} does not exist"
        cur.close()


class TestDataLoaded:
    """Verify all tables have data after bulk load."""

    def test_row_counts(self, conn):
        cur = conn.cursor()
        for table_name, expected in EXPECTED_ROW_COUNTS.items():
            cur.execute(f'SELECT count(*) FROM {table_name}')
            actual = cur.fetchone()[0]
            assert actual == expected, (
                f"{table_name}: expected {expected:,} rows, got {actual:,}"
            )
        cur.close()

    def test_total_rows(self, conn):
        cur = conn.cursor()
        total = 0
        for table_name in EXPECTED_ROW_COUNTS:
            cur.execute(f'SELECT count(*) FROM {table_name}')
            total += cur.fetchone()[0]
        cur.close()
        assert total >= 798000, f"Total rows {total:,}, expected 798,000+"

    def test_no_null_player_ids_in_weekly_stats(self, conn):
        """Garbage rows with null player_id should have been filtered during load."""
        cur = conn.cursor()
        cur.execute('SELECT count(*) FROM weekly_stats WHERE player_id IS NULL')
        null_count = cur.fetchone()[0]
        cur.close()
        assert null_count == 0, f"Found {null_count} null player_id rows in weekly_stats"


class TestIndexesExist:

    def test_key_indexes_exist(self, conn):
        cur = conn.cursor()
        expected_indexes = [
            'idx_weekly_stats_player',
            'idx_injuries_player',
            'idx_snap_counts_player',
            'idx_team_stats_team',
            'idx_games_season_week',
            'idx_players_pfr',
            'idx_players_gsis',
            'idx_ngs_passing_player',
        ]
        for idx_name in expected_indexes:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = 'public' AND indexname = %s
                )
            """, (idx_name,))
            exists = cur.fetchone()[0]
            assert exists, f"Index {idx_name} does not exist"
        cur.close()


class TestConnectionLayer:
    """Verify get_connection() and get_engine() work."""

    def test_get_connection(self):
        conn = get_connection()
        assert conn is not None
        assert conn.closed == 0
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        cur.close()
        conn.close()

    def test_get_engine(self):
        engine = get_engine()
        assert engine is not None
        result = pd.read_sql("SELECT current_database()", engine)
        assert result.iloc[0, 0] == 'nfl_predictions'

    def test_get_engine_cached(self):
        """Repeated calls return the same engine instance."""
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2


class TestQueryData:
    """Verify actual data is queryable and correct."""

    def test_mahomes_exists(self, conn):
        cur = conn.cursor()
        cur.execute("SELECT player_name, team FROM weekly_stats WHERE player_id = '00-0033873' AND season = 2024 AND week = 1")
        row = cur.fetchone()
        cur.close()
        assert row is not None
        assert row[0] == 'P.Mahomes'
        assert row[1] == 'KC'

    def test_cross_dataset_join(self, conn):
        """Join weekly_stats + games for a known player/week."""
        cur = conn.cursor()
        cur.execute("""
            SELECT ws.player_name, ws.fantasy_points_ppr, g.spread_line, g.total_line
            FROM weekly_stats ws
            JOIN games g ON ws.season = g.season AND ws.week = g.week
                AND (ws.team = g.home_team OR ws.team = g.away_team)
            WHERE ws.player_id = '00-0033873' AND ws.season = 2024 AND ws.week = 1
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        assert row is not None
        assert row[0] == 'P.Mahomes'
        assert row[2] is not None  # spread_line exists

    def test_players_id_mapping(self, conn):
        """Players table has both GSIS and PFR IDs for cross-dataset joins."""
        cur = conn.cursor()
        cur.execute("""
            SELECT gsis_id, pfr_id, display_name
            FROM players
            WHERE gsis_id = '00-0033873'
        """)
        row = cur.fetchone()
        cur.close()
        assert row is not None
        assert row[1] is not None  # pfr_id exists
        assert 'Mahomes' in row[2]


class TestModelEvalMetrics:
    """Verify the model_eval_metrics table added in Task 3.2."""

    def test_table_exists_with_expected_columns(self, conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'model_eval_metrics'
            ORDER BY ordinal_position
        """)
        cols = [row[0] for row in cur.fetchall()]
        cur.close()
        required = {"version", "position", "stat", "model_type", "mae",
                    "accuracy", "auc", "pos_class_frac", "degenerate_pob",
                    "n_eval_predictions", "n_train_rows", "algorithms",
                    "n_features", "trained_at"}
        assert required.issubset(set(cols)), \
            f"Missing columns in model_eval_metrics: {required - set(cols)}"

    def test_v5_has_54_rows(self, conn):
        """After load_model_eval.py runs, V5 should have exactly 54 rows
        (27 stat + 27 pob)."""
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM model_eval_metrics WHERE version = 'v5'")
        count = cur.fetchone()[0]
        cur.close()
        if count == 0:
            pytest.skip("V5 not loaded yet — run `python src/nfl/db/load_model_eval.py`")
        assert count == 54, f"Expected 54 V5 eval rows, found {count}"

    def test_v5_stat_pob_balance(self, conn):
        """V5 should have 27 stat + 27 pob rows."""
        cur = conn.cursor()
        cur.execute("""
            SELECT model_type, COUNT(*) FROM model_eval_metrics
            WHERE version = 'v5' GROUP BY model_type
        """)
        counts = dict(cur.fetchall())
        cur.close()
        if not counts:
            pytest.skip("V5 not loaded yet — run `python src/nfl/db/load_model_eval.py`")
        assert counts.get("stat") == 27, f"Expected 27 stat rows, got {counts.get('stat')}"
        assert counts.get("pob") == 27, f"Expected 27 pob rows, got {counts.get('pob')}"

    def test_v5_aggregate_in_model_versions(self, conn):
        """V5 aggregate row must exist in model_versions (FK prerequisite)."""
        cur = conn.cursor()
        cur.execute("""
            SELECT mae, positions FROM model_versions WHERE version = 'v5'
        """)
        row = cur.fetchone()
        cur.close()
        if row is None:
            pytest.skip("V5 not loaded yet — run `python src/nfl/db/load_model_eval.py`")
        assert row[0] is not None, "V5 aggregate MAE should not be NULL"
        assert "DST" in row[1], "V5 positions should include DST"

    def test_known_good_row_rb_rushing_yards(self, conn):
        """Spot-check a known V5 result: RB rushing_yards stat MAE ~19.99."""
        cur = conn.cursor()
        cur.execute("""
            SELECT mae, n_eval_predictions FROM model_eval_metrics
            WHERE version = 'v5' AND position = 'RB'
              AND stat = 'rushing_yards' AND model_type = 'stat'
        """)
        row = cur.fetchone()
        cur.close()
        if row is None:
            pytest.skip("V5 not loaded yet")
        assert 19.0 < row[0] < 21.0, f"RB rushing_yards MAE {row[0]} outside expected ~19.99"
        assert row[1] == 6015, f"Expected 6015 eval predictions, got {row[1]}"
