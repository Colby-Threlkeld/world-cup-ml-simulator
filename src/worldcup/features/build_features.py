"""Build the model dataset and the leakage-safe feature matrix.

Two stages:
    * :func:`build_model_dataset` reshapes the cleaned home/away matches into the
      symmetric **Team A vs Team B** modeling format (one row per match) with
      consistent targets and an explicit home-advantage flag. The same format
      serves both training and the neutral-venue 2026 simulation.
    * :func:`build_feature_matrix` (slice 3) joins leakage-safe as-of features
      (Elo, rolling form, rest days, FIFA ranking) onto that dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup.data.clean_data import (
    TARGET_CLASSES,
    TARGET_DRAW,
    TARGET_TEAM_A_WIN,
    TARGET_TEAM_B_WIN,
    TEAM_A_RESULTS,
)
from worldcup.data.validate_data import DataValidationError
from worldcup.features.rating_features import add_rating_features
from worldcup.features.rolling_features import add_rolling_features

# Column order of the Team A vs Team B model dataset.
MODEL_DATASET_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "team_a_result",
    "target_class",
    "is_neutral",
    "is_team_a_home",
    "tournament",
    "host_country",
)

_REQUIRED_MATCH_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",
    "tournament",
)


def build_model_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape cleaned home/away matches into the Team A vs Team B model dataset.

    ``team_a`` is the home team and ``team_b`` the away team — one row per match,
    never duplicated. Home advantage is preserved by ``is_team_a_home`` (True iff
    the match was *not* at a neutral venue, i.e. team A really was at home).
    Targets are derived directly from the scores, so ``team_a_result`` and
    ``target_class`` are always mutually consistent. Per-team features later join
    cleanly as ``*_a`` / ``*_b`` columns.

    Args:
        matches: Cleaned matches table from
            :func:`worldcup.data.clean_data.clean_matches`. Not mutated.

    Returns:
        The model dataset with columns :data:`MODEL_DATASET_COLUMNS`.

    Raises:
        DataValidationError: If required columns are missing.
    """
    missing = [col for col in _REQUIRED_MATCH_COLUMNS if col not in matches.columns]
    if missing:
        raise DataValidationError(
            f"matches frame missing columns for the model dataset: {missing}"
        )

    team_a_score = matches["home_score"].to_numpy()
    team_b_score = matches["away_score"].to_numpy()
    neutral = matches["neutral"].to_numpy(dtype=bool)

    a_wins = team_a_score > team_b_score
    b_wins = team_a_score < team_b_score
    team_a_result = np.select([a_wins, b_wins], ["win", "loss"], default="draw")
    target_class = np.select(
        [a_wins, b_wins], [TARGET_TEAM_A_WIN, TARGET_TEAM_B_WIN], default=TARGET_DRAW
    )

    out = pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "date": matches["date"].to_numpy(),
            "team_a": matches["home_team"].to_numpy(),
            "team_b": matches["away_team"].to_numpy(),
            "team_a_score": team_a_score,
            "team_b_score": team_b_score,
            "team_a_result": pd.Categorical(team_a_result, categories=TEAM_A_RESULTS),
            "target_class": pd.Categorical(target_class, categories=TARGET_CLASSES),
            "is_neutral": neutral,
            # team_a is the home side, so it has home advantage exactly when the
            # match is not neutral. Kept explicit so neutral-venue simulation works.
            "is_team_a_home": ~neutral,
            "tournament": matches["tournament"].astype("string").to_numpy(),
            "host_country": _host_country(matches),
        }
    )
    return out.loc[:, list(MODEL_DATASET_COLUMNS)]


def build_feature_matrix(
    matches: pd.DataFrame,
    *,
    elo_ratings: pd.DataFrame | None = None,
    fifa_rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the leakage-safe, model-ready feature matrix from cleaned matches.

    Reshapes to the Team A vs Team B model dataset, then attaches leakage-safe
    rolling team-form features (as-of the match date) for both sides plus their
    difference features. When ``elo_ratings`` and/or ``fifa_rankings`` are given,
    their as-of rating features are joined on top (see
    :func:`worldcup.features.rating_features.add_rating_features`).

    Args:
        matches: Cleaned matches table from
            :func:`worldcup.data.clean_data.clean_matches`.
        elo_ratings: Optional Elo ratings to attach (``team``, ``date``, ``elo``).
        fifa_rankings: Optional FIFA rankings to attach (``team``, ``rank_date``,
            ``rank``, ``points``).

    Returns:
        One row per match: the model dataset + rolling features (``*_a``/``*_b``)
        + difference features (+ rating features when ratings are supplied), with
        no post-kickoff information.
    """
    feats = add_rolling_features(build_model_dataset(matches))
    if elo_ratings is not None or fifa_rankings is not None:
        feats = add_rating_features(feats, elo_ratings, fifa_rankings)
    return feats


def _host_country(matches: pd.DataFrame) -> object:
    """Return the venue/host country column if available, else NA (broadcast)."""
    if "venue_country" in matches.columns:
        return matches["venue_country"].to_numpy()
    if "country" in matches.columns:
        return matches["country"].to_numpy()
    return pd.NA
