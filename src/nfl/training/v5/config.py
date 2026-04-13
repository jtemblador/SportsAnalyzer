"""V5 model training configuration.

Locks in:
- Stat-keys per position (mirrors features/v5/config.STATS_TO_PREDICT)
- Per-position algorithm subsets (decided after data-density audit)
- Per-position hyperparameters
- Count vs continuous stat classification (drives Poisson vs RMSE objective)
- Train/eval/production season splits
"""
from __future__ import annotations

from src.nfl.features.v5.config import (
    MIN_GAMES_HISTORY,
    STATS_TO_PREDICT,
)

VERSION = "v5"

# Season splits (per V5_ROADMAP.md Task 3.2)
WARMUP_SEASONS = [2018, 2019]           # rolling history only — filtered out by MIN_GAMES_HISTORY
TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024]  # available for training
EVAL_SEASONS = [2021, 2022, 2023, 2024]         # walk-forward holdout (2020 excluded — COVID)
PRODUCTION_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]  # for Task 3.2c retrain

# Per-position algorithm assignment (decided per data-density + small-data overfit risk).
POSITION_ALGORITHMS: dict[str, list[str]] = {
    "QB":  ["xgboost", "lightgbm", "catboost"],
    "RB":  ["xgboost", "lightgbm", "catboost", "random_forest"],
    "WR":  ["xgboost", "lightgbm", "catboost"],
    "TE":  ["xgboost", "lightgbm", "catboost", "random_forest"],
    "K":   ["xgboost", "random_forest"],
    "DST": ["xgboost", "catboost", "random_forest"],
}

# Per-position hyperparameters. depth and iterations tuned to position complexity + sample size.
POSITION_HYPERPARAMS: dict[str, dict] = {
    "QB":  {"depth": 9, "iterations": 200, "learning_rate": 0.05},
    "RB":  {"depth": 7, "iterations": 200, "learning_rate": 0.05},
    "WR":  {"depth": 7, "iterations": 200, "learning_rate": 0.05},
    "TE":  {"depth": 6, "iterations": 200, "learning_rate": 0.05},
    "K":   {"depth": 3, "iterations": 100, "learning_rate": 0.05},
    "DST": {"depth": 5, "iterations": 200, "learning_rate": 0.05},
}

# Count stats — modeled with Poisson loss (XGBoost count:poisson, LightGBM/CatBoost Poisson).
# Anything not in this set is continuous and uses RMSE.
COUNT_STATS: set[str] = {
    "passing_tds", "passing_interceptions", "rushing_tds", "receiving_tds",
    "receptions", "targets",
    "fg_made", "fg_att", "pat_made",
    "sacks", "interceptions", "fumble_recoveries", "defensive_tds", "safeties",
}

CONTINUOUS_STATS: set[str] = {
    "passing_yards", "rushing_yards", "receiving_yards", "points_allowed",
}

MODEL_TYPES = ["stat", "pob"]  # StatPredictor + POBModel

# LightGBM TE override — combats leaf-wise overfit on small TE sample (~3K rows)
LIGHTGBM_TE_OVERRIDES = {"min_data_in_leaf": 50}


def is_count_stat(stat: str) -> bool:
    return stat in COUNT_STATS


def get_algorithms(position: str) -> list[str]:
    if position not in POSITION_ALGORITHMS:
        raise ValueError(f"Unknown position '{position}'. Must be one of {list(POSITION_ALGORITHMS)}")
    return POSITION_ALGORITHMS[position]


def get_hyperparams(position: str) -> dict:
    if position not in POSITION_HYPERPARAMS:
        raise ValueError(f"Unknown position '{position}'. Must be one of {list(POSITION_HYPERPARAMS)}")
    return POSITION_HYPERPARAMS[position].copy()


__all__ = [
    "VERSION",
    "MIN_GAMES_HISTORY",
    "STATS_TO_PREDICT",
    "WARMUP_SEASONS", "TRAIN_SEASONS", "EVAL_SEASONS", "PRODUCTION_SEASONS",
    "POSITION_ALGORITHMS", "POSITION_HYPERPARAMS",
    "COUNT_STATS", "CONTINUOUS_STATS", "MODEL_TYPES",
    "LIGHTGBM_TE_OVERRIDES",
    "is_count_stat", "get_algorithms", "get_hyperparams",
]
