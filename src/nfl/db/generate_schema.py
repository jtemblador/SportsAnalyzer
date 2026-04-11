#!/usr/bin/env python3
"""
File: src/nfl/db/generate_schema.py

Auto-generates schema.sql from Parquet files.
Reads each dataset, maps pandas dtypes to PostgreSQL types,
and outputs CREATE TABLE statements with indexes.

Usage:
    python src/nfl/db/generate_schema.py
"""

import pandas as pd
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent.parent

# Pandas dtype -> PostgreSQL type mapping
DTYPE_MAP = {
    'int32': 'INTEGER',
    'int64': 'BIGINT',
    'float64': 'DOUBLE PRECISION',
    'object': 'TEXT',
    'bool': 'BOOLEAN',
    'datetime64[us, UTC]': 'TIMESTAMPTZ',
    'datetime64[ns, UTC]': 'TIMESTAMPTZ',
    'datetime64[ns]': 'TIMESTAMP',
}

# Table definitions: (table_name, data_dir, file_pattern_or_none, season_range, unique_constraint_columns)
# Uses ALL season files to build the column union (handles cross-season column differences)
TABLES = [
    # Reference tables (loaded first)
    ('players', 'data/nfl/players', None, None, ['gsis_id']),
    # Game-level
    ('games', 'data/nfl/schedules', 'schedules_{}.parquet', range(2018, 2026), ['game_id']),
    # Player-week (GSIS ID)
    ('weekly_stats', 'data/nfl/player_stats', 'player_stats_{}.parquet', range(2018, 2026), ['player_id', 'season', 'week']),
    ('injuries', 'data/nfl/injuries', 'injuries_{}.parquet', range(2018, 2026), None),
    ('depth_charts', 'data/nfl/depth_charts', 'depth_charts_{}.parquet', range(2018, 2025), None),
    # Player-week (PFR ID)
    ('snap_counts', 'data/nfl/snap_counts', 'snap_counts_{}.parquet', range(2018, 2026), None),
    ('pfr_pass_advstats', 'data/nfl/pfr_advstats', 'pfr_pass_{}.parquet', range(2018, 2026), None),
    ('pfr_rush_advstats', 'data/nfl/pfr_advstats', 'pfr_rush_{}.parquet', range(2018, 2026), None),
    ('pfr_rec_advstats', 'data/nfl/pfr_advstats', 'pfr_rec_{}.parquet', range(2018, 2026), None),
    # Player-week (GSIS ID, qualified players)
    ('ngs_passing', 'data/nfl/nextgen_stats', 'ngs_passing_{}.parquet', range(2018, 2026), None),
    ('ngs_rushing', 'data/nfl/nextgen_stats', 'ngs_rushing_{}.parquet', range(2018, 2026), None),
    ('ngs_receiving', 'data/nfl/nextgen_stats', 'ngs_receiving_{}.parquet', range(2018, 2026), None),
    ('ff_opportunity', 'data/nfl/ff_opportunity', 'ff_opportunity_{}.parquet', range(2018, 2026), None),
    # Team-week
    ('team_stats', 'data/nfl/team_stats', 'team_stats_{}.parquet', range(2018, 2026), ['team', 'season', 'week']),
]

# Indexes to create after tables
INDEXES = [
    # Player lookups
    ('idx_weekly_stats_player', 'weekly_stats', 'player_id, season, week'),
    ('idx_weekly_stats_team', 'weekly_stats', 'team, season, week'),
    ('idx_injuries_player', 'injuries', 'gsis_id, season, week'),
    ('idx_snap_counts_player', 'snap_counts', 'pfr_player_id, season, week'),
    ('idx_depth_charts_player', 'depth_charts', 'gsis_id, season, week'),
    # NGS
    ('idx_ngs_passing_player', 'ngs_passing', 'player_gsis_id, season, week'),
    ('idx_ngs_rushing_player', 'ngs_rushing', 'player_gsis_id, season, week'),
    ('idx_ngs_receiving_player', 'ngs_receiving', 'player_gsis_id, season, week'),
    # FF Opportunity
    ('idx_ff_opp_player', 'ff_opportunity', 'player_id, season, week'),
    # PFR
    ('idx_pfr_pass_player', 'pfr_pass_advstats', 'pfr_player_id, season, week'),
    ('idx_pfr_rush_player', 'pfr_rush_advstats', 'pfr_player_id, season, week'),
    ('idx_pfr_rec_player', 'pfr_rec_advstats', 'pfr_player_id, season, week'),
    # Team lookups
    ('idx_team_stats_team', 'team_stats', 'team, season, week'),
    ('idx_games_season_week', 'games', 'season, week'),
    # Cross-ID mapping
    ('idx_players_pfr', 'players', 'pfr_id'),
    ('idx_players_gsis', 'players', 'gsis_id'),
]


def pg_type(dtype_str):
    """Map a pandas dtype string to PostgreSQL type."""
    dtype_str = str(dtype_str)
    if dtype_str in DTYPE_MAP:
        return DTYPE_MAP[dtype_str]
    if 'datetime' in dtype_str:
        return 'TIMESTAMPTZ'
    if 'int' in dtype_str:
        return 'INTEGER'
    if 'float' in dtype_str:
        return 'DOUBLE PRECISION'
    return 'TEXT'


def generate_create_table(table_name, df, unique_cols=None):
    """Generate CREATE TABLE statement from a DataFrame."""
    lines = [f'CREATE TABLE IF NOT EXISTS {table_name} (']
    lines.append('    id SERIAL PRIMARY KEY,')

    for col in df.columns:
        col_type = pg_type(df[col].dtype)
        # Quote column names to handle reserved words
        lines.append(f'    "{col}" {col_type},')

    # Remove trailing comma from last column
    lines[-1] = lines[-1].rstrip(',')

    # Add unique constraint if specified
    if unique_cols:
        quoted = ', '.join(f'"{c}"' for c in unique_cols)
        lines[-1] += ','
        lines.append(f'    UNIQUE ({quoted})')

    lines.append(');')
    return '\n'.join(lines)


def load_union_df(data_dir, file_pattern, seasons):
    """Load all season files and return a DataFrame with the union of all columns."""
    full_dir = ROOT / data_dir

    if file_pattern is None:
        # Single file (e.g., players.parquet)
        single_file = full_dir / f"{full_dir.name}.parquet"
        if single_file.exists():
            return pd.read_parquet(single_file)
        return None

    # Load all seasons, concat to get union of columns
    dfs = []
    for season in seasons:
        filepath = full_dir / file_pattern.format(season)
        if filepath.exists():
            dfs.append(pd.read_parquet(filepath).head(1))  # Only need schema, not full data

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def generate_schema():
    """Generate the complete schema.sql file."""
    output = []
    output.append('-- NFL Predictions Database Schema')
    output.append('-- Auto-generated from Parquet files by generate_schema.py')
    output.append('-- Uses column union across all seasons to handle schema differences')
    output.append('-- Re-run to regenerate: python src/nfl/db/generate_schema.py')
    output.append('')

    # Generate CREATE TABLE for each dataset
    for table_name, data_dir, file_pattern, seasons, unique_cols in TABLES:
        df = load_union_df(data_dir, file_pattern, seasons)
        if df is None:
            print(f"  WARNING: no data found for {table_name}, skipping")
            continue

        sql = generate_create_table(table_name, df, unique_cols)
        output.append(f'-- {table_name}: {len(df.columns)} columns from {data_dir}/')
        output.append(sql)
        output.append('')
        print(f"  {table_name}: {len(df.columns)} columns")

    # Generate indexes
    output.append('')
    output.append('-- Indexes for common query patterns')
    for idx_name, table_name, columns in INDEXES:
        quoted_cols = ', '.join(f'"{c.strip()}"' for c in columns.split(','))
        output.append(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({quoted_cols});')

    schema_sql = '\n'.join(output) + '\n'

    # Write to file
    schema_path = ROOT / 'src' / 'nfl' / 'db' / 'schema.sql'
    schema_path.write_text(schema_sql)
    print(f"\n  Schema written to: {schema_path}")
    print(f"  Tables: {len(TABLES)}, Indexes: {len(INDEXES)}")

    return schema_sql


if __name__ == '__main__':
    print("Generating database schema from Parquet files...")
    generate_schema()
