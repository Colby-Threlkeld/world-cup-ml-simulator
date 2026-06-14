"""Leakage-safe rolling team-form features.

For every match, each team's form is computed from that team's matches **strictly
before** the match. Leakage is prevented structurally:

1. The team_a/team_b matches are exploded to one row per team per match, so each
   team has its full chronological history (home *and* away appearances).
2. Rows are sorted by ``(team, date, match_id)``.
3. Rolling windows are shifted by one (``shift(1)``) and look only backward, so a
   match never contributes to its own features, and any later match — being
   later in the sort — is excluded automatically.

Missing-value conventions (documented per requirement):
    * rolling averages: ``NaN`` when the team has **no** prior matches; otherwise
      the mean over the available prior matches (``min_periods=1``).
    * ``days_since_last_match``: ``NaN`` for a team's first ever match.
    * ``matches_played_last_365_days``: ``0`` for a first match (a genuine count,
      not missing).
    * difference features: ``NaN`` if either side is ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-team rolling features (computed for both team_a and team_b).
TEAM_FEATURES: tuple[str, ...] = (
    "last_5_points_per_match",
    "last_10_points_per_match",
    "last_5_goals_for_avg",
    "last_10_goals_for_avg",
    "last_5_goals_against_avg",
    "last_10_goals_against_avg",
    "last_5_goal_diff_avg",
    "last_10_goal_diff_avg",
    "days_since_last_match",
    "matches_played_last_365_days",
)

# Difference features (team_a minus team_b).
DIFFERENCE_FEATURES: tuple[str, ...] = (
    "form_5_diff",
    "form_10_diff",
    "goals_for_5_diff",
    "goals_against_5_diff",
    "goal_diff_10_diff",
    "rest_days_diff",
)

_RECENT_WINDOW_DAYS = 365
_WIN_POINTS = 3
_DRAW_POINTS = 1


def to_team_match_long(model_df: pd.DataFrame) -> pd.DataFrame:
    """Explode the team_a/team_b model dataset into one row per team per match."""
    a = pd.DataFrame(
        {
            "match_id": model_df["match_id"].to_numpy(),
            "date": model_df["date"].to_numpy(),
            "team": model_df["team_a"].to_numpy(),
            "goals_for": model_df["team_a_score"].to_numpy(),
            "goals_against": model_df["team_b_score"].to_numpy(),
        }
    )
    b = pd.DataFrame(
        {
            "match_id": model_df["match_id"].to_numpy(),
            "date": model_df["date"].to_numpy(),
            "team": model_df["team_b"].to_numpy(),
            "goals_for": model_df["team_b_score"].to_numpy(),
            "goals_against": model_df["team_a_score"].to_numpy(),
        }
    )
    long = pd.concat([a, b], ignore_index=True)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["points"] = np.select(
        [long["goals_for"] > long["goals_against"], long["goals_for"] == long["goals_against"]],
        [_WIN_POINTS, _DRAW_POINTS],
        default=0,
    )
    return long


def compute_team_features(long: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling features to the long team-match frame.

    Sorts by ``(team, date, match_id)`` and shifts by one so the current match is
    excluded; later matches sort afterward and are never seen.
    """
    long = long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)
    grp = long.groupby("team", sort=False)

    long["last_5_points_per_match"] = _rolling_mean(grp, "points", 5)
    long["last_10_points_per_match"] = _rolling_mean(grp, "points", 10)
    long["last_5_goals_for_avg"] = _rolling_mean(grp, "goals_for", 5)
    long["last_10_goals_for_avg"] = _rolling_mean(grp, "goals_for", 10)
    long["last_5_goals_against_avg"] = _rolling_mean(grp, "goals_against", 5)
    long["last_10_goals_against_avg"] = _rolling_mean(grp, "goals_against", 10)
    long["last_5_goal_diff_avg"] = _rolling_mean(grp, "goal_diff", 5)
    long["last_10_goal_diff_avg"] = _rolling_mean(grp, "goal_diff", 10)

    long["days_since_last_match"] = grp["date"].transform(lambda s: s.diff().dt.days)
    long["matches_played_last_365_days"] = grp["date"].transform(_recent_match_count)
    return long


def add_rolling_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe rolling features (``*_a``/``*_b``) and their differences.

    Args:
        model_df: Team A vs Team B model dataset from
            :func:`worldcup.features.build_features.build_model_dataset`. Must have
            ``match_id``, ``date``, ``team_a``, ``team_b``, ``team_a_score``,
            ``team_b_score``. Not mutated.

    Returns:
        A copy of ``model_df`` with each team's rolling features (suffixed ``_a``
        and ``_b``) and the difference features added.
    """
    long = compute_team_features(to_team_match_long(model_df))
    feat = long[["match_id", "team", *TEAM_FEATURES]]

    side_a = feat.rename(columns={"team": "team_a", **{f: f"{f}_a" for f in TEAM_FEATURES}})
    side_b = feat.rename(columns={"team": "team_b", **{f: f"{f}_b" for f in TEAM_FEATURES}})

    out = model_df.merge(side_a, on=["match_id", "team_a"], how="left")
    out = out.merge(side_b, on=["match_id", "team_b"], how="left")
    return _add_difference_features(out)


# --- internal helpers -------------------------------------------------------


def _rolling_mean(grp: "pd.core.groupby.DataFrameGroupBy", col: str, window: int) -> pd.Series:
    """Backward-looking rolling mean that excludes the current match (shift 1)."""
    return grp[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def _recent_match_count(dates: pd.Series) -> pd.Series:
    """Count each team's prior matches within the last 365 days (excludes current).

    Uses ``searchsorted`` on the team's ascending dates: for row ``i`` the count is
    ``i - (number of dates <= date_i - 365d)``, i.e. prior matches strictly within
    the window.
    """
    d = dates.to_numpy()
    thresholds = d - np.timedelta64(_RECENT_WINDOW_DAYS, "D")
    older_or_equal = np.searchsorted(d, thresholds, side="right")
    counts = np.arange(len(d)) - older_or_equal
    return pd.Series(np.clip(counts, 0, None), index=dates.index)


def _add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team_a-minus-team_b difference features."""
    out = df.copy()
    out["form_5_diff"] = out["last_5_points_per_match_a"] - out["last_5_points_per_match_b"]
    out["form_10_diff"] = out["last_10_points_per_match_a"] - out["last_10_points_per_match_b"]
    out["goals_for_5_diff"] = out["last_5_goals_for_avg_a"] - out["last_5_goals_for_avg_b"]
    out["goals_against_5_diff"] = (
        out["last_5_goals_against_avg_a"] - out["last_5_goals_against_avg_b"]
    )
    out["goal_diff_10_diff"] = out["last_10_goal_diff_avg_a"] - out["last_10_goal_diff_avg_b"]
    out["rest_days_diff"] = out["days_since_last_match_a"] - out["days_since_last_match_b"]
    return out
