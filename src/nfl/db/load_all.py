#!/usr/bin/env python3
"""
File: src/nfl/db/load_all.py

Bulk loads all Parquet data into the PostgreSQL database.
Reads each dataset directory, concatenates season files, and inserts
into the corresponding table via pandas to_sql().

Usage:
    python src/nfl/db/load_all.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nfl.db.connection import get_engine, get_connection

# Project root
ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'nfl'

# Table loading definitions: (table_name, data_subdir, file_pattern, season_range, filters)
# file_pattern: None = single file, str = per-season pattern
TABLES = [
    # Reference tables (first)
    ('players', 'players', None, None, None),
    # Game-level
    ('games', 'schedules', 'schedules_{}.parquet', range(2018, 2026), None),
    # Player-week (GSIS ID)
    ('weekly_stats', 'player_stats', 'player_stats_{}.parquet', range(2018, 2026), lambda df: df[df['player_id'].notna()]),
    ('injuries', 'injuries', 'injuries_{}.parquet', range(2018, 2026), None),
    ('depth_charts', 'depth_charts', 'depth_charts_{}.parquet', range(2018, 2025), None),
    # Player-week (PFR ID)
    ('snap_counts', 'snap_counts', 'snap_counts_{}.parquet', range(2018, 2026), None),
    ('pfr_pass_advstats', 'pfr_advstats', 'pfr_pass_{}.parquet', range(2018, 2026), None),
    ('pfr_rush_advstats', 'pfr_advstats', 'pfr_rush_{}.parquet', range(2018, 2026), None),
    ('pfr_rec_advstats', 'pfr_advstats', 'pfr_rec_{}.parquet', range(2018, 2026), None),
    # Player-week (GSIS ID, qualified players)
    ('ngs_passing', 'nextgen_stats', 'ngs_passing_{}.parquet', range(2018, 2026), None),
    ('ngs_rushing', 'nextgen_stats', 'ngs_rushing_{}.parquet', range(2018, 2026), None),
    ('ngs_receiving', 'nextgen_stats', 'ngs_receiving_{}.parquet', range(2018, 2026), None),
    ('ff_opportunity', 'ff_opportunity', 'ff_opportunity_{}.parquet', range(2018, 2026), None),
    # Team-week
    ('team_stats', 'team_stats', 'team_stats_{}.parquet', range(2018, 2026), None),
]


def load_table(table_name, data_subdir, file_pattern, seasons, filter_fn, engine):
    """
    Load a single table from Parquet files into PostgreSQL.

    Args:
        table_name: PostgreSQL table name
        data_subdir: Subdirectory under data/nfl/
        file_pattern: Per-season filename pattern (None = single file named after subdir)
        seasons: Range of seasons to load (None for single-file datasets)
        filter_fn: Optional function to filter DataFrame before loading
        engine: SQLAlchemy engine
    """
    data_path = DATA_DIR / data_subdir

    if file_pattern is None:
        # Single file (e.g., players.parquet)
        parquet_file = data_path / f"{data_subdir}.parquet"
        if not parquet_file.exists():
            print(f"  {table_name}: FILE NOT FOUND ({parquet_file})")
            return 0
        df = pd.read_parquet(parquet_file)
    else:
        # Per-season files
        dfs = []
        for season in seasons:
            filepath = data_path / file_pattern.format(season)
            if filepath.exists():
                dfs.append(pd.read_parquet(filepath))
        if not dfs:
            print(f"  {table_name}: NO FILES FOUND in {data_path}")
            return 0
        df = pd.concat(dfs, ignore_index=True)

    # Apply filters
    rows_before = len(df)
    if filter_fn is not None:
        df = filter_fn(df)
        filtered = rows_before - len(df)
        if filtered > 0:
            print(f"  {table_name}: filtered {filtered} rows (nulls/garbage)")

    # Load into PostgreSQL
    df.to_sql(table_name, engine, if_exists='append', index=False,
              method='multi', chunksize=5000)

    return len(df)


def load_all():
    """Load all Parquet datasets into PostgreSQL."""
    engine = get_engine()

    # Check if tables already have data
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cur.fetchall()]

    non_empty = []
    for t in tables:
        if t in [name for name, *_ in TABLES]:
            cur.execute(f"SELECT count(*) FROM {t}")
            if cur.fetchone()[0] > 0:
                non_empty.append(t)
    cur.close()
    conn.close()

    if non_empty:
        print(f"WARNING: Tables already have data: {non_empty}")
        print("Truncating all tables before loading...")
        conn = get_connection()
        cur = conn.cursor()
        for t in reversed([name for name, *_ in TABLES]):
            cur.execute(f"TRUNCATE TABLE {t} CASCADE")
        conn.commit()
        cur.close()
        conn.close()

    print("=" * 60)
    print("BULK LOADING ALL DATA INTO POSTGRESQL")
    print("=" * 60)
    print()

    total_rows = 0
    results = {}

    for table_name, data_subdir, file_pattern, seasons, filter_fn in TABLES:
        print(f"  Loading {table_name}...", end=" ")
        rows = load_table(table_name, data_subdir, file_pattern, seasons, filter_fn, engine)
        results[table_name] = rows
        total_rows += rows
        print(f"{rows:,} rows")

    print()
    print("=" * 60)
    print(f"COMPLETE: {total_rows:,} total rows across {len(results)} tables")
    print("=" * 60)
    print()

    # Summary
    for table_name, rows in results.items():
        print(f"  {table_name:<25} {rows:>10,}")

    return results


if __name__ == '__main__':
    load_all()
