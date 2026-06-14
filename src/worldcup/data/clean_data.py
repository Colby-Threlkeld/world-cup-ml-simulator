"""Clean and standardize raw match results into a tidy match table."""

from __future__ import annotations

import pandas as pd

RESULT_HOME = "H"
RESULT_DRAW = "D"
RESULT_AWAY = "A"


def match_result(home_score: int, away_score: int) -> str:
    """Return the match result from the home team's perspective.

    Args:
        home_score: Goals scored by the home team.
        away_score: Goals scored by the away team.

    Returns:
        ``"H"`` (home win), ``"D"`` (draw), or ``"A"`` (away win).
    """
    if home_score > away_score:
        return RESULT_HOME
    if home_score < away_score:
        return RESULT_AWAY
    return RESULT_DRAW


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``result`` column (H/D/A) derived from the score columns.

    Args:
        df: Match table with ``home_score`` and ``away_score`` columns.

    Returns:
        A copy of ``df`` with the ``result`` column appended.
    """
    out = df.copy()
    out["result"] = [
        match_result(int(h), int(a))
        for h, a in zip(out["home_score"], out["away_score"], strict=True)
    ]
    return out


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on raw results.

    TODO(slice 1): standardize column names, coerce dtypes, drop rows with
    missing scores, deduplicate, normalize team names via
    :mod:`worldcup.data.team_names`, sort chronologically, and attach result
    labels.

    Args:
        df: Raw results from :func:`worldcup.data.load_data.load_raw_results`.

    Returns:
        A tidy, validated match table.
    """
    raise NotImplementedError("clean_results is implemented in slice 1.")
