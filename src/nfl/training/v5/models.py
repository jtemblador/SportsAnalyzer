"""V5 ensemble models — StatPredictor (regression) + POBModel (binary classification).

Per-position algorithm subset (from config.POSITION_ALGORITHMS) and per-stat
objective (Poisson for COUNT_STATS, RMSE for continuous, LogLoss for POB).

Atomic file writes (write to .tmp, then rename) defend against Colab disconnects.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.nfl.training.v5.config import (
    LIGHTGBM_TE_OVERRIDES,
    VERSION,
    get_algorithms,
    get_hyperparams,
    is_count_stat,
)
from src.nfl.training.v5.data import NEUTRAL_FILLS, fill_features


def _make_regressor(algo: str, position: str, stat: str) -> Any:
    """Build a single regressor for the (algo, position, stat) cell."""
    hp = get_hyperparams(position)
    poisson = is_count_stat(stat)

    if algo == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            objective="count:poisson" if poisson else "reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if algo == "lightgbm":
        from lightgbm import LGBMRegressor
        kwargs = dict(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            objective="poisson" if poisson else "regression",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        if position == "TE":
            kwargs.update(LIGHTGBM_TE_OVERRIDES)
        return LGBMRegressor(**kwargs)
    if algo == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=hp["iterations"],
            depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            loss_function="Poisson" if poisson else "RMSE",
            random_seed=42,
            verbose=False,
        )
    if algo == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        # RF has no Poisson objective; falls back to MSE on integer targets.
        return RandomForestRegressor(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown algorithm '{algo}'")


def _make_classifier(algo: str, position: str) -> Any:
    """Build a single binary classifier for POBModel (always LogLoss)."""
    hp = get_hyperparams(position)

    if algo == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if algo == "lightgbm":
        from lightgbm import LGBMClassifier
        kwargs = dict(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        if position == "TE":
            kwargs.update(LIGHTGBM_TE_OVERRIDES)
        return LGBMClassifier(**kwargs)
    if algo == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=hp["iterations"],
            depth=hp["depth"],
            learning_rate=hp["learning_rate"],
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
        )
    if algo == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=hp["iterations"],
            max_depth=hp["depth"],
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown algorithm '{algo}'")


def _atomic_dump(obj: Any, path: Path) -> None:
    """Write obj to path via .tmp + rename — survives mid-write disconnect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


class _BaseEnsemble:
    model_type: str = ""

    def __init__(self, position: str, stat: str):
        self.position = position
        self.stat = stat
        self.algorithms = get_algorithms(position)
        self.models: dict[str, Any] = {}
        self.feature_columns: list[str] = []

    def _build_estimator(self, algo: str) -> Any:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns = list(X.columns)
        # Fill NaN with 0 — tree models handle 0 cleanly; avoids per-algo divergence
        # on missing-value handling (e.g., XGBoost native vs RF needing imputation).
        X_filled = fill_features(X)
        for algo in self.algorithms:
            est = self._build_estimator(algo)
            est.fit(X_filled, y)
            self.models[algo] = est

    def _predict_per_algo(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        # Must use fill_features (not raw fillna) to match train-time imputation.
        # Skipping NEUTRAL_FILLS here would re-introduce the dome-temp bias the
        # NEUTRAL_FILLS policy was created to eliminate.
        X_aligned = fill_features(X[self.feature_columns])
        return {algo: est.predict(X_aligned) for algo, est in self.models.items()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        per_algo = self._predict_per_algo(X)
        stack = np.stack(list(per_algo.values()), axis=0)
        # Use nanmean so a single NaN-producing algo doesn't poison the row,
        # but assert at least one algo produced a valid prediction per row.
        out = np.nanmean(stack, axis=0)
        if np.isnan(out).any():
            n_bad = int(np.isnan(out).sum())
            raise RuntimeError(
                f"All-algo NaN predictions for {n_bad} rows in "
                f"{self.position}/{self.stat}/{self.model_type} — "
                f"every algorithm returned NaN. Check input data for degenerate values."
            )
        return out

    def save(self, models_dir: Path, algorithms_used: list[str] | None = None,
             extra_meta: dict | None = None) -> Path:
        """Save one .joblib per algo + one shared meta JSON. Atomic."""
        models_dir = Path(models_dir)
        algos = algorithms_used or self.algorithms
        for algo in algos:
            if algo not in self.models:
                continue
            fname = f"{self.position}_{self.stat}_{self.model_type}_{algo}.joblib"
            _atomic_dump(self.models[algo], models_dir / fname)
        meta_path = models_dir / f"{self.position}_{self.stat}_{self.model_type}_meta.json"
        meta = {
            "version": VERSION,
            "position": self.position,
            "stat": self.stat,
            "model_type": self.model_type,
            "algorithms_used": algos,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "objective_per_algo": self._objective_per_algo(),
            "hyperparameters": get_hyperparams(self.position),
            # Embed imputation policy so Task 3.2c can detect drift between the
            # policy this model was trained under and the current data.NEUTRAL_FILLS.
            "neutral_fills": dict(NEUTRAL_FILLS),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)
        _atomic_write_json(meta, meta_path)
        return meta_path

    def _objective_per_algo(self) -> dict[str, str]:
        raise NotImplementedError


class StatPredictor(_BaseEnsemble):
    """Regression ensemble — predicts raw stat value."""
    model_type = "stat"

    def _build_estimator(self, algo: str) -> Any:
        return _make_regressor(algo, self.position, self.stat)

    def _objective_per_algo(self) -> dict[str, str]:
        poisson = is_count_stat(self.stat)
        out = {}
        for algo in self.algorithms:
            if algo == "random_forest":
                out[algo] = "mse"  # RF has no Poisson — documented bias
            elif poisson:
                out[algo] = "poisson"
            else:
                out[algo] = "rmse"
        return out


class POBModel(_BaseEnsemble):
    """Binary classifier — P(actual > rolling_avg_<stat>). Always LogLoss."""
    model_type = "pob"

    def _build_estimator(self, algo: str) -> Any:
        return _make_classifier(algo, self.position)

    def _objective_per_algo(self) -> dict[str, str]:
        return {algo: "logloss" for algo in self.algorithms}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return mean probability of the positive class across algos."""
        X_aligned = fill_features(X[self.feature_columns])
        probs = []
        for est in self.models.values():
            p = est.predict_proba(X_aligned)[:, 1]
            probs.append(p)
        out = np.nanmean(np.stack(probs, axis=0), axis=0)
        if np.isnan(out).any():
            n_bad = int(np.isnan(out).sum())
            raise RuntimeError(
                f"All-algo NaN POB probabilities for {n_bad} rows in "
                f"{self.position}/{self.stat} — every classifier returned NaN."
            )
        return out


def ensemble_files_complete(models_dir: Path, position: str, stat: str,
                            model_type: str, algorithms: list[str]) -> bool:
    """Resume check — returns True iff all algo .joblib + meta JSON exist."""
    models_dir = Path(models_dir)
    meta_path = models_dir / f"{position}_{stat}_{model_type}_meta.json"
    if not meta_path.exists():
        return False
    for algo in algorithms:
        p = models_dir / f"{position}_{stat}_{model_type}_{algo}.joblib"
        if not p.exists():
            return False
    return True


__all__ = [
    "StatPredictor",
    "POBModel",
    "ensemble_files_complete",
]
