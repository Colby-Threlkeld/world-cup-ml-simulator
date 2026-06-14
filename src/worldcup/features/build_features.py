"""Build the model dataset and the leakage-safe feature matrix.

Three stages:
    * :func:`build_model_dataset` reshapes the cleaned home/away matches into the
      symmetric **Team A vs Team B** modeling format (one row per match) with
      consistent targets and an explicit home-advantage flag. The same format
      serves both training and the neutral-venue 2026 simulation.
    * :func:`build_feature_matrix` joins leakage-safe as-of features (rolling
      form, rest days, and — when supplied — Elo / FIFA ranking) onto it.
    * The IO/validation helpers (:func:`load_matches`, :func:`load_optional_ratings`,
      :func:`validate_feature_matrix`, :func:`missing_value_summary`,
      :func:`save_features`) back the ``scripts/build_features.py`` pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup.config import PROCESSED_DIR
from worldcup.data.clean_data import (
    TARGET_CLASSES,
    TARGET_DRAW,
    TARGET_TEAM_A_WIN,
    TARGET_TEAM_B_WIN,
    TEAM_A_RESULTS,
)
from worldcup.data.validate_data import DataValidationError
from worldcup.features.rating_features import add_rating_features
from worldcup.features.rolling_features import (
    DIFFERENCE_FEATURES,
    TEAM_FEATURES,
    add_rolling_features,
)

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = PROCESSED_DIR / "features.csv"

# Rolling feature columns the matrix must carry (both sides + their differences).
ROLLING_FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"{feat}_{side}" for side in ("a", "b") for feat in TEAM_FEATURES),
    *DIFFERENCE_FEATURES,
)

# Label columns — derived from the scores, never to be used as model inputs.
LABEL_COLUMNS: tuple[str, ...] = (
    "team_a_score",
    "team_b_score",
    "team_a_result",
    "target_class",
)

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
    # Sort by match_id so the written table is deterministic regardless of any
    # upstream row ordering.
    return feats.sort_values("match_id", kind="stable").reset_index(drop=True)


def _host_country(matches: pd.DataFrame) -> object:
    """Return the venue/host country column if available, else NA (broadcast)."""
    if "venue_country" in matches.columns:
        return matches["venue_country"].to_numpy()
    if "country" in matches.columns:
        return matches["country"].to_numpy()
    return pd.NA


# --- pipeline IO / validation helpers ---------------------------------------


def load_matches(path: Path | str) -> pd.DataFrame:
    """Load a cleaned matches table from CSV or parquet, ready for feature build.

    Parses ``date`` to datetime, coerces ``neutral`` to bool, and casts the score
    columns to a nullable integer dtype so downstream reshaping is well-typed.

    Args:
        path: Path to ``matches.csv`` (or a ``.parquet`` file).

    Returns:
        The matches frame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If required columns are missing.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"matches file not found: {src}")

    matches = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    missing = [col for col in _REQUIRED_MATCH_COLUMNS if col not in matches.columns]
    if missing:
        raise DataValidationError(f"matches file missing required columns: {missing}")

    matches["date"] = pd.to_datetime(matches["date"])
    matches["neutral"] = _coerce_bool(matches["neutral"])
    for col in ("home_score", "away_score"):
        matches[col] = matches[col].astype("int64")
    return matches


def load_optional_ratings(
    path: Path | str | None,
    *,
    date_col: str,
    label: str,
) -> pd.DataFrame | None:
    """Load an optional ratings CSV, returning ``None`` (with a warning) if absent.

    The pipeline must not crash when a rating source is missing — it simply skips
    that feature block. A *present but malformed* file is a real error and still
    raises.

    Args:
        path: Path to the ratings CSV, or ``None`` to skip.
        date_col: The frame's effective-date column (parsed to datetime).
        label: Human name for log messages (e.g. ``"Elo ratings"``).

    Returns:
        The parsed ratings frame, or ``None`` if the file is absent.

    Raises:
        DataValidationError: If the file exists but lacks ``date_col``.
    """
    if path is None:
        logger.warning("No %s file provided; skipping those features.", label)
        return None
    src = Path(path)
    if not src.exists():
        logger.warning("%s file not found at %s; skipping those features.", label, src)
        return None

    ratings = pd.read_csv(src)
    if date_col not in ratings.columns:
        raise DataValidationError(
            f"{label} file {src} missing date column '{date_col}'"
        )
    ratings[date_col] = pd.to_datetime(ratings[date_col])
    logger.info("Loaded %d %s row(s) from %s", len(ratings), label, src)
    return ratings


def validate_feature_matrix(
    features: pd.DataFrame, *, expected_rows: int | None = None
) -> None:
    """Validate the assembled feature table before it is written.

    Checks structural integrity and leakage-relevant invariants: required columns
    present, ``match_id`` unique, no missing keys/labels, and (optionally) one row
    per input match. Rolling/rating features are *allowed* to be NaN (genuine
    "no prior history"), so those are summarized, not rejected.

    Args:
        features: The assembled feature matrix.
        expected_rows: If given, the feature table must have exactly this many
            rows (one per played match — no duplication or drops).

    Raises:
        DataValidationError: If any structural check fails (all failures reported
            together).
    """
    errors: list[str] = []

    required = (*MODEL_DATASET_COLUMNS, *ROLLING_FEATURE_COLUMNS)
    missing = [col for col in required if col not in features.columns]
    if missing:
        # Without the key columns the remaining checks would raise KeyError.
        raise DataValidationError(f"feature matrix missing columns: {missing}")

    if expected_rows is not None and len(features) != expected_rows:
        errors.append(
            f"row count {len(features)} != expected {expected_rows} (one per match)"
        )

    n_dup_ids = int(features["match_id"].duplicated().sum())
    if n_dup_ids:
        errors.append(f"{n_dup_ids} duplicate match_id value(s)")

    key_and_label = ("match_id", "date", "team_a", "team_b", *LABEL_COLUMNS)
    null_counts = features[list(key_and_label)].isna().sum()
    nulls = {col: int(n) for col, n in null_counts.items() if n > 0}
    if nulls:
        errors.append(f"missing values in key/label columns: {nulls}")

    if not features["match_id"].is_monotonic_increasing:
        errors.append("match_id is not sorted ascending (non-deterministic order)")

    if errors:
        raise DataValidationError(
            "\n".join(
                [f"{len(errors)} validation error(s) in feature matrix:", *(f"  - {e}" for e in errors)]
            )
        )


def missing_value_summary(features: pd.DataFrame) -> pd.Series:
    """Return per-column NaN counts (descending), for logging data coverage."""
    counts = features.isna().sum()
    return counts[counts > 0].sort_values(ascending=False)


def save_features(features: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Write the feature table to CSV (deterministic), creating parent dirs.

    Args:
        features: The assembled feature matrix.
        path: Output path. Defaults to ``data/processed/features.csv``.

    Returns:
        The path written to.
    """
    out_path = Path(path) if path is not None else DEFAULT_FEATURES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    logger.info("Wrote %d feature rows to %s", len(features), out_path)
    return out_path


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a bool-or-text column (``True``/``false``/``1``/``0``) to real bool."""
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    coerced = series.astype("string").str.strip().str.lower().map(mapping)
    if coerced.isna().any():
        bad = list(series[coerced.isna()].unique()[:5])
        raise DataValidationError(f"unparseable boolean values: {bad}")
    return coerced.astype(bool)
