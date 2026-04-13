"""V5 training data loader, history filter, POB target builder.

Per-position parquet loading (player vs DST), feature/target separation,
and leakage prevention by dropping current-week raw stat columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.nfl.training.v5.config import (
    COUNT_STATS,
    MIN_GAMES_HISTORY,
)

# All current-week raw-stat columns to drop from features (leakage prevention).
# Anything observable only AFTER the game ends cannot be a feature.
PLAYER_RAW_STAT_COLS = {
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds", "carries",
    "receptions", "receiving_yards", "receiving_tds", "targets",
    "completions", "attempts",
    "fg_made", "fg_att", "pat_made",
    "fantasy_points", "fantasy_points_ppr",
    "passing_epa", "rushing_epa", "receiving_epa",
}

DST_RAW_STAT_COLS = {
    "sacks", "interceptions", "fumble_recoveries", "defensive_tds",
    "safeties", "points_allowed", "blocked_kicks", "return_tds",
}

PLAYER_IDENTITY_COLS = {
    "player_id", "player_name", "position", "team", "season", "week",
    "opponent_team", "season_type",
}
DST_IDENTITY_COLS = {
    "team", "opponent_team", "season", "week", "season_type", "is_home",
}


def _features_dir() -> Path:
    return Path("data/nfl/features/v5")


def load_features(
    position: str,
    seasons: Iterable[int],
    features_dir: Path | None = None,
) -> pd.DataFrame:
    """Load V5 features for a single position across seasons.

    DST reads features_dst_{season}.parquet; everything else reads
    features_{season}.parquet (filtered to position).
    """
    base = features_dir or _features_dir()
    found: list[tuple[int, pd.DataFrame]] = []  # keep (season, df) paired
    for season in seasons:
        if position == "DST":
            path = base / f"features_dst_{season}.parquet"
        else:
            path = base / f"features_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if position != "DST":
            df = df[df["position"] == position].copy()
        found.append((season, df))
    if not found:
        raise FileNotFoundError(
            f"No feature files found for position={position} in {base}"
        )

    # Defend against silent column drift across seasons — if a 2024 parquet has
    # a column 2020 lacks, concat fills with NaN, which then `fillna(0.0)` silently
    # imputes as 0 across 4 historical seasons. Warn so we know.
    # Pair (season, columns) directly so a missing-from-disk season can't shift
    # the warning's season attribution.
    col_sets = [(s, frozenset(f.columns)) for s, f in found]
    unique_col_sets = {cs for _, cs in col_sets}
    if len(unique_col_sets) > 1:
        all_cols: set = set().union(*unique_col_sets)
        import warnings
        for s, cs in col_sets:
            extra = sorted(all_cols - cs)
            if extra:
                warnings.warn(
                    f"Season {s} parquet missing {len(extra)} columns present in "
                    f"other seasons: {extra[:10]}{'...' if len(extra) > 10 else ''}. "
                    f"These will be NaN-filled across {s}'s rows.",
                    stacklevel=2,
                )

    frames = [f for _, f in found]
    out = pd.concat(frames, ignore_index=True)
    # Defend against silent dtype drift — string season would make _strict_prior_mask
    # produce all-False, silently emptying every training fold.
    if not pd.api.types.is_integer_dtype(out["season"]):
        out["season"] = out["season"].astype(int)
    if not pd.api.types.is_integer_dtype(out["week"]):
        out["week"] = out["week"].astype(int)
    return out


def apply_history_filter(df: pd.DataFrame, min_games: int = MIN_GAMES_HISTORY) -> pd.DataFrame:
    """Drop rows where games_of_history < min_games."""
    if "games_of_history" not in df.columns:
        raise KeyError("games_of_history column missing — was features built with V5?")
    mask = df["games_of_history"] >= min_games
    return df.loc[mask].reset_index(drop=True)


# Whitelist of safe game-context columns observable BEFORE kickoff (Vegas lines,
# weather, rest, derived game script). These are not historical aggregates but
# are still leak-free because they are set pre-game.
SAFE_PREGAME_CONTEXT = {
    "spread_line", "total_line", "team_implied_total", "opponent_implied_total",
    "team_rest", "opponent_rest", "div_game",
    "team_implied_total_diff", "opponent_implied_total_diff",
    "temp", "wind", "game_script_index", "games_of_history",
    # Pre-game published reports (depth chart and injury status are public before kickoff)
    "injury_severity",
}

# Safe column prefixes — historical aggregates and pre-game context.
SAFE_PREFIXES = ("rolling_", "variance_", "trend_", "opp_", "prior_", "is_")

# Per-column NaN imputation (applied before fillna(0)). Only columns where
# 0 is a wrong/misleading default get an explicit value:
#   - temp: 36% of rows are domes (NaN). Filling with 0 trains "freezing" on
#     1/3 of all games. 65°F is a neutral dome/mild outdoor proxy.
#   - wind: 36% NaN for the same dome rows. 0 mph is already correct semantically
#     (controlled environment), so no special handling needed.
# Rolling/variance/trend/is_ columns stay at 0 — represents "no prior signal"
# or "false flag" which is the right interpretation.
NEUTRAL_FILLS: dict[str, float] = {
    "temp": 65.0,
}


def fill_features(X: "pd.DataFrame") -> "pd.DataFrame":
    """Apply per-column NaN imputation, then 0 for everything else.

    Centralizes the imputation policy so train-time and predict-time agree.
    """
    out = X.copy()
    for col, val in NEUTRAL_FILLS.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out.fillna(0.0)


def get_feature_columns(df: pd.DataFrame, position: str) -> list[str]:
    """Return the safe feature column list (whitelist approach).

    Whitelist prevents data leakage from current-week observable stats
    (e.g., target_share, wopr, ngs_*, pfr_*, *_exp, *_diff) which appear
    raw in the V5 features parquet alongside their rolling_* equivalents.
    Only historical aggregates (rolling_/variance_/trend_/opp_/prior_) and
    pre-game context (Vegas, weather, rest) are allowed as features.

    Also filters to numeric/bool dtypes only — XGBoost rejects object columns.
    """
    candidates = []
    for c in df.columns:
        if c.startswith(SAFE_PREFIXES) or c in SAFE_PREGAME_CONTEXT:
            candidates.append(c)
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])]


def drop_all_null_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return the subset of feature_cols that have at least one non-null value.

    Used for K position to combat curse of dimensionality (600 rows × 238 cols
    where most cols are 100% null because K rows lack passing/receiving stats).
    """
    keep = [c for c in feature_cols if df[c].notna().any()]
    return keep


def prepare_stat_predictor_data(
    df: pd.DataFrame,
    stat: str,
    position: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare (X, y, feature_cols) for StatPredictor training.

    Drops rows with NaN target. Asserts non-negative for COUNT_STATS (Poisson safety).
    """
    if stat not in df.columns:
        raise KeyError(f"Target stat '{stat}' not in DataFrame columns")
    df = df.dropna(subset=[stat]).reset_index(drop=True)

    if stat in COUNT_STATS:
        min_val = df[stat].min()
        if min_val < 0:
            raise ValueError(
                f"COUNT_STATS '{stat}' has negative values (min={min_val}); "
                f"Poisson objective requires non-negative targets."
            )

    feature_cols = get_feature_columns(df, position)
    if position == "K":
        feature_cols = drop_all_null_columns(df, feature_cols)

    X = df[feature_cols]
    y = df[stat].astype(float)
    return X, y, feature_cols


def prepare_pob_data(
    df: pd.DataFrame,
    stat: str,
    position: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare (X, y_binary, feature_cols) for POBModel training.

    Drops rows where rolling_avg_<stat> is NaN (insufficient baseline) AND rows
    with NaN target. Binary label: 1 if actual > rolling_avg_<stat>, else 0.
    """
    baseline_col = f"rolling_avg_{stat}"
    if baseline_col not in df.columns:
        raise KeyError(f"Baseline column '{baseline_col}' not in DataFrame")
    if stat not in df.columns:
        raise KeyError(f"Target stat '{stat}' not in DataFrame columns")

    df = df.dropna(subset=[stat, baseline_col]).reset_index(drop=True)

    feature_cols = get_feature_columns(df, position)
    if position == "K":
        feature_cols = drop_all_null_columns(df, feature_cols)

    X = df[feature_cols]
    y = (df[stat] > df[baseline_col]).astype(int)
    return X, y, feature_cols


def attach_keys(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Return identity-key columns aligned to df's index — used by walk-forward to
    record predictions back to (season, week, player|team) tuples.

    For DST: returned `team` column maps to `player_id = team_abbr` in the
    predictions table (see V5_ROADMAP.md Task 3.2c, line ~502). Task 3.2c's
    DB-load step must perform this rename when inserting DST rows.
    """
    if position == "DST":
        keep = ["season", "week", "team", "opponent_team"]
    else:
        keep = ["season", "week", "player_id", "player_name", "team", "position"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


__all__ = [
    "load_features",
    "apply_history_filter",
    "get_feature_columns",
    "drop_all_null_columns",
    "fill_features",
    "prepare_stat_predictor_data",
    "prepare_pob_data",
    "attach_keys",
    "PLAYER_RAW_STAT_COLS",
    "DST_RAW_STAT_COLS",
    "PLAYER_IDENTITY_COLS",
    "DST_IDENTITY_COLS",
    "NEUTRAL_FILLS",
    "SAFE_PREGAME_CONTEXT",
    "SAFE_PREFIXES",
]
