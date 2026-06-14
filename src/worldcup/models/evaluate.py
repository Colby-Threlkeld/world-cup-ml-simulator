"""Evaluate match-probability models with proper scoring rules.

Probabilistic forecasts are judged with proper scoring rules (Brier score, log
loss) on a temporally held-out set -- never random K-fold, which would leak
future information into the past.
"""

from __future__ import annotations

import pandas as pd


def temporal_backtest(features: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    """Backtest a model by training before ``cutoff`` and scoring after it.

    TODO(slice 2/5): walk forward through time, fit on matches before each
    cutoff, predict the next window, and aggregate Brier score, log loss, and
    accuracy. This is the single source of truth for "is model B better than
    the Elo baseline?".

    Args:
        features: Feature matrix with labels.
        cutoff: The train/test split date.

    Returns:
        Mapping of metric name to value (e.g. ``{"brier": ..., "log_loss": ...}``).
    """
    raise NotImplementedError("temporal_backtest is implemented in slice 2.")
