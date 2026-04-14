"""V5 training orchestrator.

For each (position, stat, model_type) ensemble:
  1. Skip if all algo .joblib + meta JSON already exist (resumable).
  2. Load features for position across TRAIN_SEASONS.
  3. Apply history filter.
  4. Walk-forward eval over EVAL_SEASONS → predictions DataFrame.
  5. Compute MAE / accuracy / AUC.
  6. Re-fit on full filtered training set, save .joblib + meta.
  7. Append summary row to _mae_summary.csv (atomic).

Run:
    python -m src.nfl.training.v5.train                  # all positions
    python -m src.nfl.training.v5.train --positions DST  # one position
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nfl.training.v5.config import (
    EVAL_SEASONS,
    MODEL_TYPES,
    STATS_TO_PREDICT,
    TRAIN_SEASONS,
    VERSION,
    get_algorithms,
)
from src.nfl.training.v5.data import (
    apply_history_filter,
    load_features,
    prepare_pob_data,
    prepare_stat_predictor_data,
)
from src.nfl.training.v5.models import (
    POBModel,
    StatPredictor,
    ensemble_files_complete,
)
from src.nfl.training.v5.walkforward import (
    compute_mae,
    compute_pob_metrics,
    walk_forward_eval,
)


def _models_dir() -> Path:
    return Path("data/nfl/models") / VERSION


def _summary_path() -> Path:
    return _models_dir() / "_mae_summary.csv"


# Canonical known columns across all stat/pob rows. New keys outside this set
# signal a real schema change (e.g., an intentional column rename in a future
# V5.1) and trigger rotation. stat↔pob transitions do NOT trigger rotation
# because both are legitimate row shapes — pandas unions their columns with NaN.
_KNOWN_SUMMARY_COLUMNS: set[str] = {
    # shared
    "version", "position", "stat", "model_type", "algorithms",
    "n_train_rows", "n_features", "n_eval_predictions", "status", "trained_at",
    # stat-only
    "mae_v5",
    # pob-only
    "accuracy", "auc", "pos_class_frac", "degenerate_pob",
}


def _atomic_append_csv(row: dict, path: Path) -> None:
    """Append one row to CSV. Read-modify-write with atomic replace.

    Schema handling:
    - stat rows have `mae_v5` but no accuracy/auc; pob rows have the opposite.
      Both shapes are valid — pandas concat unions columns with NaN fills,
      which is what Task 3.2b ablation expects (filters by model_type).
    - Rotation ONLY triggers if an incoming row introduces a column name NOT
      in _KNOWN_SUMMARY_COLUMNS (e.g., a future schema evolution). This
      prevents the spurious rotation-on-every-transition bug from Round 3.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    unexpected_cols = set(row.keys()) - _KNOWN_SUMMARY_COLUMNS
    if path.exists() and unexpected_cols:
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%Y%m%dT%H%M%S")
        rotated = path.with_name(f"{path.stem}_pre_schema_{stamp}.csv")
        os.replace(path, rotated)
        import warnings
        warnings.warn(
            f"_mae_summary.csv schema drift — rotated old file to {rotated.name} "
            f"(unexpected cols: {sorted(unexpected_cols)}). Starting fresh.",
            stacklevel=2,
        )
        df = pd.DataFrame([row])
    elif path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def train_one_ensemble(
    position: str,
    stat: str,
    model_type: str,
    features_df: pd.DataFrame,
    models_dir: Path,
) -> dict:
    """Train one (position, stat, model_type) ensemble end-to-end. Returns summary row."""
    algos = get_algorithms(position)

    if model_type == "stat":
        model_factory = lambda: StatPredictor(position, stat)
    else:
        model_factory = lambda: POBModel(position, stat)

    # 1. Walk-forward eval
    preds_df = walk_forward_eval(
        model_factory=model_factory,
        df=features_df,
        position=position,
        stat=stat,
        eval_seasons=EVAL_SEASONS,
        model_type=model_type,
    )

    # 2. Refit on full filtered set, save
    if model_type == "stat":
        try:
            X, y, _ = prepare_stat_predictor_data(features_df, stat, position)
        except (KeyError, ValueError) as e:
            return {"position": position, "stat": stat, "model_type": model_type,
                    "status": f"skip_prep_error:{e}"}
    else:
        try:
            X, y, _ = prepare_pob_data(features_df, stat, position)
        except KeyError as e:
            return {"position": position, "stat": stat, "model_type": model_type,
                    "status": f"skip_prep_error:{e}"}

    if len(X) < 50:
        return {"position": position, "stat": stat, "model_type": model_type,
                "status": f"skip_insufficient_rows:{len(X)}"}

    model = model_factory()
    model.fit(X, y)  # type: ignore[attr-defined]

    # Per-ensemble metrics. Column named `mae_v5` (not `mae`) per the schema
    # contract with Task 3.2b — ablation joins this on (position, stat) to
    # diff vs ablated runs.
    if model_type == "stat":
        mae = compute_mae(preds_df)
        eval_metrics = {"mae_v5": mae, "n_eval_predictions": len(preds_df)}
    else:
        m = compute_pob_metrics(preds_df)
        # Cast bool→int so CSV round-trip preserves truthiness (pandas writes
        # bool as "True"/"False" string, which evaluates truthy on read-back).
        eval_metrics = {"accuracy": m["accuracy"], "auc": m["auc"],
                        "pos_class_frac": m["pos_class_frac"],
                        "degenerate_pob": int(bool(m.get("degenerate_pob", False))),
                        "n_eval_predictions": len(preds_df)}

    extra_meta = {
        "n_train_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "eval_seasons": list(EVAL_SEASONS),
        "eval_metrics": eval_metrics,
    }
    model.save(models_dir, algorithms_used=algos, extra_meta=extra_meta)  # type: ignore[attr-defined]

    summary = {
        "version": VERSION,
        "position": position,
        "stat": stat,
        "model_type": model_type,
        "algorithms": ",".join(algos),
        "n_train_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_eval_predictions": int(len(preds_df)),
        "status": "ok",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **eval_metrics,
    }
    return summary


def train_all(
    positions: list[str] | None = None,
    seasons: list[int] | None = None,
    models_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Train all (or filtered) ensembles. Resumable per-ensemble."""
    positions = positions or list(STATS_TO_PREDICT.keys())
    seasons = seasons or list(TRAIN_SEASONS)
    models_dir = models_dir or _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Sweep orphaned .tmp files from prior interrupted runs. Atomic write
    # always ends with os.replace(tmp, final), so any .tmp remaining = died
    # mid-write = stale partial joblib that would be overwritten anyway.
    orphans = list(models_dir.glob("*.tmp"))
    if orphans:
        print(f"  Cleaning up {len(orphans)} orphaned .tmp files from prior interrupted run(s)")
        for o in orphans:
            try:
                o.unlink()
            except OSError:
                pass

    summary_rows = []
    for position in positions:
        print(f"\n=== Loading features for {position} ===")
        try:
            df = load_features(position, seasons)
        except FileNotFoundError as e:
            print(f"  SKIP {position}: {e}")
            continue
        df = apply_history_filter(df)
        print(f"  {len(df)} rows after history filter")

        algos = get_algorithms(position)
        for stat in STATS_TO_PREDICT[position]:
            for model_type in MODEL_TYPES:
                if not force and ensemble_files_complete(
                    models_dir, position, stat, model_type, algos
                ):
                    print(f"  SKIP {position}/{stat}/{model_type} — complete files exist")
                    continue
                print(f"  TRAIN {position}/{stat}/{model_type} ({len(algos)} algos)")
                row = train_one_ensemble(position, stat, model_type, df, models_dir)
                summary_rows.append(row)
                _atomic_append_csv(row, _summary_path())
                if row.get("status") == "ok":
                    if model_type == "stat":
                        print(f"    MAE={row.get('mae_v5'):.3f} on {row.get('n_eval_predictions')} preds")
                    else:
                        print(f"    AUC={row.get('auc'):.3f} acc={row.get('accuracy'):.3f} "
                              f"pos_frac={row.get('pos_class_frac'):.2f}")
                else:
                    print(f"    {row['status']}")

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", nargs="*", default=None,
                        help="Subset of positions (default: all 6)")
    parser.add_argument("--seasons", nargs="*", type=int, default=None,
                        help="Override training seasons")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if complete files exist")
    args = parser.parse_args()
    train_all(positions=args.positions, seasons=args.seasons, force=args.force)


if __name__ == "__main__":
    main()
