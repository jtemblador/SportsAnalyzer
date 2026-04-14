#!/usr/bin/env python3
"""
File: src/nfl/db/load_model_eval.py

Loads V5 (or any future version's) per-ensemble evaluation metrics into the
`model_eval_metrics` table, plus a single aggregate row into `model_versions`.

Source: data/nfl/models/v5/_mae_summary_consolidated.csv (54 rows: 27 stat + 27 pob).

Usage:
    python src/nfl/db/load_model_eval.py                   # load V5 (default)
    python src/nfl/db/load_model_eval.py --version v5_ablated_rolling \\
        --csv data/nfl/models/v5_ablated_rolling/_mae_summary_consolidated.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nfl.db.connection import get_connection

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "data/nfl/models/v5/_mae_summary_consolidated.csv"
DEFAULT_VERSION = "v5"

# PPR scoring weights for computing weighted-average aggregate MAE
# (matches scripts/analyze_v5_vs_v4.py). Used to produce ONE fantasy-points MAE
# for model_versions.mae so V5 can be compared to V1-V4 numerically.
PPR_WEIGHTS = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": 2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "fg_made": 3.0, "fg_att": 0.0, "pat_made": 1.0,
}


def compute_aggregate_mae(df: pd.DataFrame) -> float:
    """Weighted-sum fantasy-points MAE across skill positions (upper bound
    estimate, matches scripts/analyze_v5_vs_v4.py).

    Returns the 70%-of-upper realistic estimate (empirical correlation factor).
    DST excluded from aggregate because V1-V4 didn't predict DST.
    """
    stat_rows = df[(df["model_type"] == "stat") & (df["position"] != "DST")]
    upper_totals = {}
    for pos in stat_rows["position"].unique():
        sub = stat_rows[stat_rows["position"] == pos]
        total = 0.0
        for _, row in sub.iterrows():
            w = PPR_WEIGHTS.get(row["stat"])
            if w is None:
                continue
            total += w * row["mae_v5"]
        upper_totals[pos] = total
    if not upper_totals:
        return float("nan")
    avg_upper = sum(upper_totals.values()) / len(upper_totals)
    return avg_upper * 0.7  # realistic estimate


def upsert_model_version(cur, version: str, description: str,
                         aggregate_mae: float, seasons_str: str,
                         positions_str: str) -> None:
    """Insert or update a row in model_versions. Required by FK from
    model_eval_metrics."""
    cur.execute(
        """
        INSERT INTO model_versions (version, description, mae, prediction_weeks,
                                    prediction_season, positions)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (version) DO UPDATE SET
            description = EXCLUDED.description,
            mae = EXCLUDED.mae,
            prediction_weeks = EXCLUDED.prediction_weeks,
            prediction_season = EXCLUDED.prediction_season,
            positions = EXCLUDED.positions
        """,
        (version, description, aggregate_mae, seasons_str, 2024, positions_str),
    )


def delete_existing_eval_rows(cur, version: str) -> int:
    """Clear out any prior rows for this version so re-runs are idempotent."""
    cur.execute(
        "DELETE FROM model_eval_metrics WHERE version = %s", (version,)
    )
    return cur.rowcount


def insert_eval_rows(cur, version: str, df: pd.DataFrame) -> int:
    """Insert 54 rows into model_eval_metrics from the consolidated CSV."""
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO model_eval_metrics
                (version, position, stat, model_type, mae, accuracy, auc,
                 pos_class_frac, degenerate_pob, n_eval_predictions,
                 n_train_rows, algorithms, n_features)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                version,
                row["position"],
                row["stat"],
                row["model_type"],
                None if pd.isna(row.get("mae_v5")) else float(row["mae_v5"]),
                None if pd.isna(row.get("accuracy")) else float(row["accuracy"]),
                None if pd.isna(row.get("auc")) else float(row["auc"]),
                None if pd.isna(row.get("pos_class_frac")) else float(row["pos_class_frac"]),
                int(row.get("degenerate_pob", 0) or 0),
                None if pd.isna(row.get("n_eval_predictions")) else int(row["n_eval_predictions"]),
                None if pd.isna(row.get("n_train_rows")) else int(row["n_train_rows"]),
                row.get("algorithms", ""),
                None if pd.isna(row.get("n_features")) else int(row["n_features"]),
            ),
        )
    return len(df)


def load_model_eval(csv_path: Path, version: str,
                    description: str, seasons_str: str,
                    positions_str: str) -> None:
    df = pd.read_csv(csv_path)
    assert len(df) == 54, f"Expected 54 rows in CSV, got {len(df)}"
    print(f"Loaded {len(df)} rows from {csv_path}")

    aggregate_mae = compute_aggregate_mae(df)
    print(f"Computed aggregate MAE (realistic fantasy PPR estimate): {aggregate_mae:.3f}")

    conn = get_connection()
    cur = conn.cursor()
    try:
        upsert_model_version(cur, version, description, aggregate_mae,
                             seasons_str, positions_str)
        print(f"Upserted {version} into model_versions")

        deleted = delete_existing_eval_rows(cur, version)
        if deleted:
            print(f"Deleted {deleted} existing eval rows for {version}")

        inserted = insert_eval_rows(cur, version, df)
        print(f"Inserted {inserted} eval rows")

        conn.commit()
        print("Committed.")
    except Exception:
        conn.rollback()
        print("ROLLED BACK on error")
        raise
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="Path to _mae_summary_consolidated.csv")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help="Model version tag for model_versions.version")
    parser.add_argument("--description", default=(
        "V5 — whitelist features, Poisson for counts, walk-forward eval, "
        "per-position algos, DST added"))
    parser.add_argument("--seasons", default="2021-2024",
                        help="Eval season range string")
    parser.add_argument("--positions", default="QB,RB,WR,TE,K,DST")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    load_model_eval(
        csv_path=csv_path,
        version=args.version,
        description=args.description,
        seasons_str=args.seasons,
        positions_str=args.positions,
    )


if __name__ == "__main__":
    main()
