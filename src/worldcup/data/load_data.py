"""Load raw datasets (international results + FIFA rankings) into DataFrames.

Readers are intentionally thin: they read bytes and check the column schema but
do not clean or transform — that is :mod:`worldcup.data.clean_data`'s job, so the
raw load stays reproducible and side-effect free.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from worldcup.config import RAW_DIR
from worldcup.data.validate_data import DataValidationError

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.csv"
RANKINGS_FILENAME = "fifa_ranking.csv"

# Columns the martj42 results.csv is expected to provide (verified by the audit).
RAW_MATCH_COLUMNS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
)


def load_raw_matches(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw international match results CSV.

    Args:
        path: Explicit path to the CSV. Defaults to ``data/raw/results.csv``.

    Returns:
        Raw, unvalidated matches — one row per match, scores possibly null for
        unplayed future fixtures.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If any expected column is missing.
    """
    csv_path = Path(path) if path is not None else (RAW_DIR / RESULTS_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the results dataset into data/raw/ "
            "(see README) or pass an explicit path."
        )
    logger.info("Loading raw matches from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [col for col in RAW_MATCH_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"{csv_path.name} is missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    logger.info("Loaded %d raw match rows", len(df))
    return df


def load_raw_rankings(path: Path | str | None = None) -> pd.DataFrame:
    """Load the FIFA ranking CSV (V1.1 — not used by the V1 pipeline yet).

    Args:
        path: Explicit path. Defaults to ``data/raw/fifa_ranking.csv``.

    Returns:
        One row per team per ranking-release date (raw, unvalidated).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    csv_path = Path(path) if path is not None else (RAW_DIR / RANKINGS_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. The FIFA ranking dataset is a V1.1 addition."
        )
    logger.info("Loading raw FIFA rankings from %s", csv_path)
    return pd.read_csv(csv_path, encoding="utf-8")
