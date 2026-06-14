"""Leakage-safe rating features via as-of joins.

Attaches each team's strength rating — self-computed **Elo** and the **FIFA
ranking** (rank + points) — onto the Team A vs Team B model dataset. Every value
is the most recent rating *known before kickoff*; a rating dated after the match
can never be used.

Leakage is prevented structurally by :func:`pandas.merge_asof` with
``direction="backward"``: for each match it picks the latest rating row whose
effective date is ``<= match date``. A rating dated even one day after the match
sorts later and is skipped. :func:`validate_rating_features` re-checks this after
the join as a belt-and-suspenders guard.

Expected input schemas (extra columns are ignored):
    * ``elo_ratings``: ``team``, ``date``, ``elo`` — one row per team per
      effective date, holding the team's **pre-match** Elo as of ``date``.
    * ``fifa_rankings``: ``team``, ``rank_date``, ``rank``, ``points`` — one row
      per team per FIFA ranking release.

Missing-value conventions (documented per requirement):
    * A team with **no** rating dated at/before the match gets ``NaN`` for that
      rating (and rank/points).
    * Difference features (``*_diff``) are ``NaN`` if either side is ``NaN``.
"""

from __future__ import annotations

import pandas as pd

from worldcup.data.team_names import normalize_team_name
from worldcup.data.validate_data import DataValidationError, validate_rating_features

# Elo rating features (team_a, team_b, and their difference).
ELO_FEATURES: tuple[str, ...] = ("team_a_elo", "team_b_elo", "elo_diff")

# FIFA ranking features (rank and points per side, plus differences).
FIFA_FEATURES: tuple[str, ...] = (
    "team_a_fifa_rank",
    "team_b_fifa_rank",
    "fifa_rank_diff",
    "team_a_fifa_points",
    "team_b_fifa_points",
    "fifa_points_diff",
)

# Every rating feature this module can attach.
RATING_FEATURES: tuple[str, ...] = (*ELO_FEATURES, *FIFA_FEATURES)

# Internal columns holding the chosen rating's effective date (for the leakage
# re-check); dropped from the returned frame.
_ELO_DATE_COLS: tuple[str, ...] = ("team_a_elo_date", "team_b_elo_date")
_FIFA_DATE_COLS: tuple[str, ...] = ("team_a_fifa_date", "team_b_fifa_date")

_REQUIRED_MODEL_COLS: tuple[str, ...] = ("match_id", "date", "team_a", "team_b")


def add_rating_features(
    model_df: pd.DataFrame,
    elo_ratings: pd.DataFrame | None = None,
    fifa_rankings: pd.DataFrame | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Attach leakage-safe Elo and FIFA-ranking features to the model dataset.

    For each match, both sides receive the most recent rating dated at or before
    the match date (as-of join). Team names in the ratings frames are normalized
    first so a spelling mismatch never silently drops a rating.

    Args:
        model_df: Team A vs Team B model dataset (needs ``match_id``, ``date``,
            ``team_a``, ``team_b``). Not mutated.
        elo_ratings: Elo ratings (``team``, ``date``, ``elo``). If ``None``, Elo
            features are skipped.
        fifa_rankings: FIFA rankings (``team``, ``rank_date``, ``rank``,
            ``points``). If ``None``, FIFA features are skipped.
        validate: If ``True``, re-check after joining that no chosen rating is
            dated after its match (raises :class:`LeakageError` otherwise).

    Returns:
        A copy of ``model_df`` with the requested rating features added.

    Raises:
        DataValidationError: If ``model_df`` is missing required columns or a
            ratings frame is missing its expected columns.
        LeakageError: If ``validate`` and any joined rating post-dates its match.
    """
    missing = [c for c in _REQUIRED_MODEL_COLS if c not in model_df.columns]
    if missing:
        raise DataValidationError(
            f"model_df missing columns for rating features: {missing}"
        )

    out = model_df.copy()
    date_cols: list[str] = []

    if elo_ratings is not None:
        out = _join_elo(out, elo_ratings)
        out["elo_diff"] = out["team_a_elo"] - out["team_b_elo"]
        date_cols.extend(_ELO_DATE_COLS)

    if fifa_rankings is not None:
        out = _join_fifa(out, fifa_rankings)
        out["fifa_rank_diff"] = out["team_a_fifa_rank"] - out["team_b_fifa_rank"]
        out["fifa_points_diff"] = out["team_a_fifa_points"] - out["team_b_fifa_points"]
        date_cols.extend(_FIFA_DATE_COLS)

    if validate and date_cols:
        validate_rating_features(out, date_cols, match_date_col="date")

    return out.drop(columns=date_cols)


# --- internal helpers -------------------------------------------------------


def _join_elo(model_df: pd.DataFrame, elo_ratings: pd.DataFrame) -> pd.DataFrame:
    """As-of join Elo onto both sides; adds value + (internal) snapshot date."""
    ratings = _prepare_ratings(
        elo_ratings, required=("team", "date", "elo"), date_col="date"
    )
    out = model_df
    for side in ("a", "b"):
        out = _asof_join_side(
            out,
            ratings,
            team_col=f"team_{side}",
            ratings_date_col="date",
            value_map={"elo": f"team_{side}_elo"},
            date_out=f"team_{side}_elo_date",
        )
    return out


def _join_fifa(model_df: pd.DataFrame, fifa_rankings: pd.DataFrame) -> pd.DataFrame:
    """As-of join FIFA rank + points onto both sides (one snapshot per side)."""
    ratings = _prepare_ratings(
        fifa_rankings,
        required=("team", "rank_date", "rank", "points"),
        date_col="rank_date",
    )
    out = model_df
    for side in ("a", "b"):
        out = _asof_join_side(
            out,
            ratings,
            team_col=f"team_{side}",
            ratings_date_col="rank_date",
            value_map={
                "rank": f"team_{side}_fifa_rank",
                "points": f"team_{side}_fifa_points",
            },
            date_out=f"team_{side}_fifa_date",
        )
    return out


def _prepare_ratings(
    ratings: pd.DataFrame,
    *,
    required: tuple[str, ...],
    date_col: str,
) -> pd.DataFrame:
    """Validate, normalize team names, and sort a ratings frame for ``merge_asof``."""
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise DataValidationError(f"ratings frame missing columns: {missing}")

    out = ratings[list(required)].copy()
    out["team"] = [normalize_team_name(str(t)) for t in out["team"]]
    out[date_col] = pd.to_datetime(out[date_col])
    # merge_asof requires the right frame globally sorted on the join key.
    return out.sort_values(date_col, kind="stable").reset_index(drop=True)


def _asof_join_side(
    model_df: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    team_col: str,
    ratings_date_col: str,
    value_map: dict[str, str],
    date_out: str,
) -> pd.DataFrame:
    """Backward as-of join one side's ratings; returns a copy with new columns.

    Picks, per match, the latest rating row whose date is ``<= match date``
    (``allow_exact_matches=True`` keeps a same-day rating, since these frames hold
    pre-match snapshots). Result is realigned to ``model_df`` by ``match_id``.
    """
    left = (
        model_df[["match_id", "date", team_col]]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    right = ratings.rename(columns={**value_map, ratings_date_col: date_out})

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=date_out,
        left_by=team_col,
        right_by="team",
        direction="backward",
        allow_exact_matches=True,
    )

    keep = ["match_id", *value_map.values(), date_out]
    return model_df.merge(merged[keep], on="match_id", how="left")
