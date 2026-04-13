# src/nfl/features/v5/utils.py
"""Shared helpers used by both player-week and team-week rolling pipelines."""

import numpy as np
import pandas as pd
from src.nfl.features.v5.config import ROLLING_DECAY, ROLLING_WINDOW


def decay_weighted_avg(values, decay=ROLLING_DECAY):
    """Decay-weighted average of a chronological sequence (oldest first).
    Most-recent values get the highest weight (decay**0 = 1)."""
    if len(values) == 0:
        return np.nan
    rev = values[::-1]
    weights = np.array([decay ** i for i in range(len(rev))])
    return np.sum(rev * weights) / np.sum(weights)


# ---------------------------------------------------------------------------
# Position-safe rolling helpers — designed for use with df.groupby(...).transform.
# transform() preserves the original DataFrame index regardless of group
# iteration order, eliminating the latent silent-corruption risk of the
# list-append + bulk-assign pattern. Math is byte-for-byte identical to the
# prior implementation: same window slice [max(0, i-window):i], same NaN
# filter, same decay function.
# ---------------------------------------------------------------------------

def rolling_decay_avg_series(s: pd.Series, window: int = ROLLING_WINDOW,
                             decay: float = ROLLING_DECAY) -> pd.Series:
    """Per-row decay-weighted average over the prior `window` values
    (strictly before current row, NaN-filtered). Returns a Series aligned
    to the input's index."""
    arr = s.to_numpy()
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        past = arr[max(0, i - window):i]
        past = past[~pd.isna(past)]
        if len(past) > 0:
            out[i] = decay_weighted_avg(past, decay=decay)
    return pd.Series(out, index=s.index)


def rolling_variance_series(s: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Per-row std-dev over the prior `window` values (NaN if < 2 prior values).
    Note: column name in callers is `variance_*` but the value is std-dev —
    pre-existing convention, kept for compatibility."""
    arr = s.to_numpy()
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        past = arr[max(0, i - window):i]
        past = past[~pd.isna(past)]
        if len(past) >= 2:
            out[i] = np.std(past)
    return pd.Series(out, index=s.index)


def rolling_trend_series(s: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Per-row trend = (mean of recent 3 prior - mean of older prior) / older.
    Returns NaN if < 4 prior values, 0.0 if older mean == 0."""
    arr = s.to_numpy()
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        past = arr[max(0, i - window):i]
        past = past[~pd.isna(past)]
        if len(past) >= 4:
            recent = np.mean(past[-3:])
            older = np.mean(past[:-3])
            out[i] = (recent - older) / older if older != 0 else 0.0
    return pd.Series(out, index=s.index)
