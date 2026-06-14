"""Clean and standardize raw match results into a tidy, validated match table.

The output is the ``matches`` table from the data audit: one row per *played*
match with canonical team names, integer scores, result labels, and a stable
``match_id``. Unplayed future fixtures (null scores — e.g. the already-drawn 2026
World Cup games) are split out here and never reach the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup.config import INTERIM_DIR
from worldcup.data.load_data import RAW_MATCH_COLUMNS
from worldcup.data.team_names import normalize_team_columns
from worldcup.data.validate_data import DataValidationError, validate_matches

logger = logging.getLogger(__name__)

RESULT_HOME = "H"
RESULT_DRAW = "D"
RESULT_AWAY = "A"

DEFAULT_MATCHES_PATH = INTERIM_DIR / "matches.parquet"

# Final column order of the cleaned matches table.
MATCHES_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",
    "tournament",
    "is_competitive",
    "result",
    "home_win",
    "draw",
    "away_win",
    "total_goals",
    "goal_diff",
    "venue_city",
    "venue_country",
)


def match_result(home_score: int, away_score: int) -> str:
    """Return the match result from the home team's perspective ("H"/"D"/"A")."""
    if home_score > away_score:
        return RESULT_HOME
    if home_score < away_score:
        return RESULT_AWAY
    return RESULT_DRAW


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``result`` column (H/D/A) derived from the score columns (returns a copy)."""
    out = df.copy()
    out["result"] = [
        match_result(int(h), int(a))
        for h, a in zip(out["home_score"], out["away_score"], strict=True)
    ]
    return out


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw results into the tidy ``matches`` table (played matches only).

    Pipeline: parse dates -> drop exact duplicate rows -> assign ``match_id`` over
    the full sorted set -> drop unplayed (null-score) fixtures -> validate & cast
    scores -> normalize team names -> standardize tournament -> coerce ``neutral``
    -> rename venue columns -> add result labels and derived columns.

    Args:
        df: Raw results, e.g. from :func:`worldcup.data.load_data.load_raw_matches`.

    Returns:
        A new cleaned DataFrame; the input is never mutated.

    Raises:
        DataValidationError: On empty input, missing columns, unparseable dates,
            or invalid scores.
    """
    if df.empty:
        raise DataValidationError("cannot clean an empty matches frame")
    _require_columns(df, RAW_MATCH_COLUMNS)

    out = _parse_dates(df)
    out = _drop_duplicate_rows(out)

    out = out.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    # match_id is assigned over the full sorted set (played + unplayed) so ids stay
    # stable when a fixture is later played and re-ingested.
    out["match_id"] = np.arange(len(out), dtype="int64")

    out, n_unplayed = _drop_unplayed(out)
    if n_unplayed:
        logger.info("Dropped %d unplayed fixture row(s) with null scores", n_unplayed)

    out = _validate_and_cast_scores(out)
    out = normalize_team_columns(out, ["home_team", "away_team"])

    out["tournament"] = out["tournament"].astype("string").str.strip()
    out["is_competitive"] = (out["tournament"] != "Friendly").astype(bool)
    out["tournament"] = out["tournament"].astype("category")
    out["neutral"] = _coerce_neutral(out["neutral"])
    out = out.rename(columns={"city": "venue_city", "country": "venue_country"})

    out = _add_result_labels(out)
    out["total_goals"] = (out["home_score"] + out["away_score"]).astype("int16")
    out["goal_diff"] = (out["home_score"] - out["away_score"]).astype("int16")

    out = out.loc[:, list(MATCHES_COLUMNS)]
    validate_matches(out)  # final integrity gate (reuses the schema validator)
    logger.info("Cleaned matches: %d played rows x %d columns", len(out), out.shape[1])
    return out


def save_matches(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Write the cleaned matches table to parquet, creating parent dirs.

    Args:
        df: Cleaned matches table.
        path: Output path. Defaults to ``data/interim/matches.parquet``.

    Returns:
        The path written to.
    """
    out_path = Path(path) if path is not None else DEFAULT_MATCHES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d matches to %s", len(df), out_path)
    return out_path


# --- internal helpers -------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"matches frame missing required columns: {missing}")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``date`` column as YYYY-MM-DD, raising clearly on bad values."""
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    unparseable = parsed.isna() & df["date"].notna()
    if unparseable.any():
        sample = df.loc[unparseable, "date"].head(5).tolist()
        raise DataValidationError(
            f"{int(unparseable.sum())} unparseable date(s); expected YYYY-MM-DD, e.g. {sample}"
        )
    out = df.copy()
    out["date"] = parsed
    return out


def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact full-row duplicates only (keeps legitimate same-day rematches)."""
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    dropped = n_before - len(out)
    if dropped:
        logger.info("Dropped %d exact duplicate row(s)", dropped)
    return out


def _drop_unplayed(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Split off unplayed fixtures (null scores); return (played, n_unplayed)."""
    played = df["home_score"].notna() & df["away_score"].notna()
    out = df.loc[played].reset_index(drop=True)
    return out, int((~played).sum())


def _validate_and_cast_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure scores are non-negative integers, then cast to int16."""
    out = df.copy()
    for col in ("home_score", "away_score"):
        scores = out[col]
        if (scores % 1 != 0).any():
            raise DataValidationError(f"column '{col}' has non-integer scores")
        if (scores < 0).any():
            raise DataValidationError(f"column '{col}' has negative scores")
    out["home_score"] = out["home_score"].astype("int16")
    out["away_score"] = out["away_score"].astype("int16")
    return out


def _coerce_neutral(series: pd.Series) -> pd.Series:
    """Coerce the ``neutral`` column to a real bool, accepting bool or true/false text."""
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    coerced = series.astype("string").str.strip().str.lower().map(mapping)
    if coerced.isna().any():
        bad = list(series[coerced.isna()].unique()[:5])
        raise DataValidationError(f"unparseable 'neutral' values: {bad}")
    return coerced.astype(bool)


def _add_result_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add result (H/D/A) plus the home_win / draw / away_win boolean labels."""
    out = df.copy()
    home, away = out["home_score"], out["away_score"]
    out["home_win"] = (home > away).astype(bool)
    out["draw"] = (home == away).astype(bool)
    out["away_win"] = (home < away).astype(bool)
    out["result"] = pd.Categorical(
        np.select([home > away, home < away], [RESULT_HOME, RESULT_AWAY], default=RESULT_DRAW),
        categories=[RESULT_HOME, RESULT_DRAW, RESULT_AWAY],
    )
    return out
