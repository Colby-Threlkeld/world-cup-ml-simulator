"""Leakage-safe rolling team-form features.

Every rolling statistic for a match must be computed from matches strictly
*before* that match's date. We enforce this with a one-step ``shift`` so a match
can never contribute to its own features -- the cardinal sin of football
modeling.
"""

from __future__ import annotations

import pandas as pd


def add_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    out_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add a leakage-safe rolling mean of ``value_col`` within each group.

    For each row the statistic uses up to ``window`` prior observations for that
    group, ordered by ``date_col``, and **excludes the current row**.

    Args:
        df: Long-format table (typically one row per team per match).
        group_col: Column to group by (e.g. team).
        value_col: Numeric column to average (e.g. goals scored).
        window: Maximum number of prior matches to include.
        out_col: Name of the output column to add.
        date_col: Column used to order matches chronologically.

    Returns:
        A copy of ``df`` sorted by ``date_col`` with ``out_col`` added. The value
        is ``NaN`` for rows that have no prior history.
    """
    out = df.sort_values(date_col).copy()
    out[out_col] = out.groupby(group_col)[value_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return out
