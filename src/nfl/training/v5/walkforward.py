"""Walk-forward expanding-window evaluation.

For each (eval_season, eval_week), train on every row with (season, week) <
(eval_season, eval_week), predict that week. Yields predictions for honest
out-of-sample MAE/AUC reporting.

Mirrors how the model is used in production (predict next week given everything
prior) — not a fixed train/test chunk.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.nfl.training.v5.data import (
    attach_keys,
    prepare_pob_data,
    prepare_stat_predictor_data,
)


def _strict_prior_mask(df: pd.DataFrame, eval_season: int, eval_week: int) -> pd.Series:
    """Boolean mask: True where (season, week) < (eval_season, eval_week)."""
    return (df["season"] < eval_season) | (
        (df["season"] == eval_season) & (df["week"] < eval_week)
    )


def _eval_iter(df: pd.DataFrame, eval_seasons: Iterable[int]) -> list[tuple[int, int]]:
    """List of (season, week) pairs to evaluate, sorted ascending."""
    sub = df[df["season"].isin(list(eval_seasons))]
    pairs = sorted(set(zip(sub["season"].tolist(), sub["week"].tolist())))
    return pairs


def walk_forward_eval(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    position: str,
    stat: str,
    eval_seasons: Iterable[int],
    model_type: str,
    min_train_rows: int = 50,
) -> pd.DataFrame:
    """Run walk-forward eval for one (position, stat, model_type) ensemble.

    model_factory: callable returning a fresh ensemble instance per fold
                   (e.g., lambda: StatPredictor('QB', 'passing_yards'))
    Returns DataFrame: identity keys + actual + predicted columns.
    """
    if model_type not in {"stat", "pob"}:
        raise ValueError(f"model_type must be 'stat' or 'pob', got '{model_type}'")

    prepare = prepare_stat_predictor_data if model_type == "stat" else prepare_pob_data
    pred_col = "predicted" if model_type == "stat" else "probability_over"
    actual_col = "actual" if model_type == "stat" else "exceeded_baseline"

    rows = []
    for season, week in _eval_iter(df, eval_seasons):
        train_mask = _strict_prior_mask(df, season, week)
        train_df = df.loc[train_mask].copy()
        eval_df = df.loc[(df["season"] == season) & (df["week"] == week)].copy()

        if len(train_df) < min_train_rows or len(eval_df) == 0:
            continue

        # Apply the same row-drop the prepare function will apply, so identity
        # keys stay aligned with X_eval / y_eval / preds. This prevents an
        # index-misalignment hazard (Task 3.1.5-class bug).
        baseline_col = f"rolling_avg_{stat}"
        eval_drop_cols = [stat] if model_type == "stat" else [stat, baseline_col]
        eval_df_aligned = eval_df.dropna(subset=eval_drop_cols).reset_index(drop=True)

        # Prepare both splits with same prep function (handles target dropping etc.)
        try:
            X_train, y_train, _ = prepare(train_df, stat, position)
            X_eval, y_eval, _ = prepare(eval_df_aligned, stat, position)
        except KeyError:
            continue
        except ValueError as exc:
            # ValueError = bad data (e.g., negative COUNT_STAT) → don't silently
            # nuke the whole fold. Surface so feature-engineering issues are visible.
            import warnings
            warnings.warn(
                f"walk_forward_eval skipping {position}/{stat}/{model_type} "
                f"fold {season}w{week}: {exc}",
                stacklevel=2,
            )
            continue

        if len(X_train) < min_train_rows or len(X_eval) == 0:
            continue

        # Align eval feature columns to training feature set (K may differ across folds
        # if drop_all_null_columns prunes differently).
        train_features = list(X_train.columns)
        for c in train_features:
            if c not in X_eval.columns:
                X_eval[c] = 0.0
        X_eval = X_eval[train_features]

        model = model_factory()
        model.fit(X_train, y_train)  # type: ignore[attr-defined]
        preds = model.predict(X_eval)  # type: ignore[attr-defined]

        # eval_df_aligned, X_eval, y_eval, preds all have matching length now.
        keys = attach_keys(eval_df_aligned, position).reset_index(drop=True)
        if len(keys) != len(preds):
            # Defensive: should never trigger after the alignment fix above.
            raise RuntimeError(
                f"Eval-key/prediction length mismatch in {position}/{stat}/{model_type}: "
                f"{len(keys)} keys vs {len(preds)} preds"
            )
        fold = keys.assign(
            stat=stat,
            model_type=model_type,
            **{actual_col: y_eval.values, pred_col: preds},
        )
        rows.append(fold)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_mae(predictions_df: pd.DataFrame) -> float:
    """MAE for StatPredictor predictions (requires 'actual' + 'predicted' cols)."""
    if predictions_df.empty:
        return float("nan")
    err = (predictions_df["predicted"] - predictions_df["actual"]).abs()
    return float(err.mean())


def compute_pob_metrics(predictions_df: pd.DataFrame) -> dict:
    """Accuracy + AUC for POB predictions.

    Sets `degenerate_pob=True` when class balance is so skewed (<5% or >95%)
    that accuracy is meaningless (a constant predictor would score >95%).
    Downstream summary should treat such rows as untrustworthy.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    if predictions_df.empty:
        return {"accuracy": float("nan"), "auc": float("nan"),
                "pos_class_frac": float("nan"), "degenerate_pob": False}
    y = predictions_df["exceeded_baseline"].values
    p = predictions_df["probability_over"].values
    pred_class = (p > 0.5).astype(int)
    pos_frac = float(np.mean(y))
    metrics: dict = {
        "accuracy": float(accuracy_score(y, pred_class)),
        "pos_class_frac": pos_frac,
        "degenerate_pob": pos_frac < 0.05 or pos_frac > 0.95,
    }
    try:
        metrics["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        metrics["auc"] = float("nan")  # only one class present
    return metrics


__all__ = [
    "walk_forward_eval",
    "compute_mae",
    "compute_pob_metrics",
    "_strict_prior_mask",
    "_eval_iter",
]
