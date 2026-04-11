#!/usr/bin/env python3
"""
File: src/nfl/db/load_predictions.py

Loads all prediction Parquet files into the predictions table and
backfills actual values from weekly_stats.

Usage:
    python src/nfl/db/load_predictions.py
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nfl.db.connection import get_engine, get_connection

ROOT = Path(__file__).parent.parent.parent.parent
PREDICTIONS_DIR = ROOT / 'data' / 'nfl' / 'predictions'

# Model version metadata (hardcoded — these are historical facts)
MODEL_VERSIONS = [
    ('v1_baseline_mae5.14', 'V1 Baseline — rolling averages, basic features', 5.14, '2-12', 2025, 'QB,RB,WR,TE,K'),
    ('v2_variance_trends_mae4.66', 'V2 — variance, usage trends, opponent context', 4.66, '1-13', 2025, 'QB,RB,WR,TE,K'),
    ('v3_epa_efficiency', 'V3 — EPA, efficiency metrics (no MAE improvement)', 4.66, '1-13', 2025, 'QB,RB,WR,TE,K'),
    ('v4_position_specific', 'V4 — Vegas odds, position-specific hyperparameters', 4.26, '1-14', 2025, 'QB,RB,WR,TE,K'),
]

# Mapping from prediction stat names to weekly_stats column names
STAT_COLUMN_MAP = {
    'fantasy_points_ppr': 'fantasy_points_ppr',
    'passing_yards': 'passing_yards',
    'passing_tds': 'passing_tds',
    'passing_interceptions': 'passing_interceptions',
    'rushing_yards': 'rushing_yards',
    'rushing_tds': 'rushing_tds',
    'receiving_yards': 'receiving_yards',
    'receiving_tds': 'receiving_tds',
    'receptions': 'receptions',
    'fg_made': 'fg_made',
    'fg_att': 'fg_att',
}


def load_model_versions(engine):
    """Insert model version metadata. Call after truncating predictions."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM model_versions")
    for version, desc, mae, weeks, season, positions in MODEL_VERSIONS:
        cur.execute(
            "INSERT INTO model_versions (version, description, mae, prediction_weeks, prediction_season, positions) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (version, desc, mae, weeks, season, positions)
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"  Loaded {len(MODEL_VERSIONS)} model versions")


def load_prediction_files(engine):
    """Load all prediction Parquet files into the predictions table."""
    total_rows = 0
    for version, _, _, _, _, _ in MODEL_VERSIONS:
        version_dir = PREDICTIONS_DIR / version
        if not version_dir.exists():
            print(f"  {version}: directory not found, skipping")
            continue

        parquet_files = sorted(version_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  {version}: no parquet files, skipping")
            continue

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            df['version'] = version
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Ensure consistent column order for to_sql
        columns = [
            'version', 'player_id', 'player_name', 'position', 'team',
            'opponent', 'season', 'week', 'stat', 'model_type',
            'predicted_value', 'predicted_diff', 'confidence_lower',
            'confidence_upper', 'baseline', 'probability_over',
        ]
        combined = combined[columns]

        combined.to_sql('predictions', engine, if_exists='append',
                        index=False, method='multi', chunksize=5000)
        total_rows += len(combined)
        print(f"  {version}: {len(combined):,} rows ({len(parquet_files)} files)")

    return total_rows


def backfill_actuals(engine):
    """
    Backfill actual_value and error columns by joining with weekly_stats.
    Each prediction stat maps to a column in weekly_stats.
    """
    conn = get_connection()
    cur = conn.cursor()

    total_updated = 0
    for stat, ws_column in STAT_COLUMN_MAP.items():
        cur.execute(f"""
            UPDATE predictions p
            SET actual_value = ws."{ws_column}",
                error = ABS(p.predicted_value - ws."{ws_column}")
            FROM weekly_stats ws
            WHERE p.player_id = ws.player_id
              AND p.season = ws.season
              AND p.week = ws.week
              AND p.stat = %s
              AND p.predicted_value IS NOT NULL
        """, (stat,))
        updated = cur.rowcount
        total_updated += updated

    conn.commit()
    cur.close()
    conn.close()
    return total_updated


def load_all_predictions():
    """Full ETL: load model versions, predictions, and backfill actuals."""
    engine = get_engine()

    print("=" * 60)
    print("LOADING PREDICTIONS INTO POSTGRESQL")
    print("=" * 60)
    print()

    # Truncate predictions first (FK references model_versions)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE predictions RESTART IDENTITY")
    conn.commit()
    cur.close()
    conn.close()

    print("Loading model versions...")
    load_model_versions(engine)
    print()

    print("Loading prediction files...")
    total_rows = load_prediction_files(engine)
    print(f"\n  Total: {total_rows:,} prediction rows loaded")
    print()

    print("Backfilling actual values from weekly_stats...")
    updated = backfill_actuals(engine)
    print(f"  Updated {updated:,} rows with actual values")
    print()

    # Summary
    df = pd.read_sql(text("""
        SELECT version,
               COUNT(*) as total_predictions,
               COUNT(actual_value) as with_actuals,
               ROUND(AVG(error)::numeric, 2) as avg_error
        FROM predictions
        WHERE predicted_value IS NOT NULL
        GROUP BY version
        ORDER BY version
    """), engine)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    return total_rows


if __name__ == '__main__':
    load_all_predictions()
