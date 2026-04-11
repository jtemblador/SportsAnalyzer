"""
Tests for the PostgreSQL database schema — verifies database exists,
all tables are created, and column counts match expected values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import psycopg2
import pandas as pd
from src.nfl.db.config import DB_CONFIG
from src.nfl.db.connection import get_connection, get_engine

# Expected column counts (Parquet columns + 1 for auto-generated id)
EXPECTED_TABLES = {
    'players': 40,
    'games': 49,
    'weekly_stats': 115,
    'injuries': 17,
    'depth_charts': 16,
    'snap_counts': 17,
    'pfr_pass_advstats': 25,
    'pfr_rush_advstats': 17,
    'pfr_rec_advstats': 18,
    'ngs_passing': 30,
    'ngs_rushing': 23,
    'ngs_receiving': 24,
    'ff_opportunity': 160,
    'team_stats': 103,
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

    def test_all_14_tables_exist(self, conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        cur.close()
        assert len(tables) == 14, f"Expected 14 tables, found {len(tables)}: {tables}"

    def test_each_table_exists(self, conn):
        cur = conn.cursor()
        for table_name in EXPECTED_TABLES:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (table_name,))
            exists = cur.fetchone()[0]
            assert exists, f"Table {table_name} does not exist"
        cur.close()


class TestColumnCounts:

    def test_column_counts_match(self, conn):
        cur = conn.cursor()
        for table_name, expected_cols in EXPECTED_TABLES.items():
            cur.execute("""
                SELECT count(*) FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table_name,))
            actual = cur.fetchone()[0]
            assert actual == expected_cols, (
                f"{table_name}: expected {expected_cols} columns, got {actual}"
            )
        cur.close()


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


class TestTablesEmpty:
    """Verify tables exist but have no data yet (data loading is Task 1.3)."""

    def test_all_tables_empty(self, conn):
        cur = conn.cursor()
        for table_name in EXPECTED_TABLES:
            cur.execute(f'SELECT count(*) FROM {table_name}')
            count = cur.fetchone()[0]
            assert count == 0, f"{table_name} has {count} rows (expected 0)"
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
        engine.dispose()

    def test_get_engine_read_table(self):
        """Engine can read from an actual table (even if empty)."""
        engine = get_engine()
        result = pd.read_sql("SELECT count(*) as cnt FROM players", engine)
        assert result.iloc[0, 0] == 0
        engine.dispose()
