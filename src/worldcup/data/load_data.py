"""Load raw datasets (international results + FIFA rankings) into DataFrames.

These readers are intentionally thin: they only read bytes from disk and parse
dates. All cleaning, validation, and normalization happen downstream so the raw
load stays reproducible and side-effect free.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.config import RAW_DIR

RESULTS_FILENAME = "results.csv"
RANKINGS_FILENAME = "fifa_ranking.csv"


def load_raw_results(path: Path | None = None) -> pd.DataFrame:
    """Load the international match results CSV.

    Args:
        path: Explicit path to the CSV. Defaults to ``data/raw/results.csv``.

    Returns:
        One row per international match (raw, unvalidated).

    Raises:
        FileNotFoundError: If the file does not exist (download it in slice 1).
    """
    csv_path = path or (RAW_DIR / RESULTS_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the results dataset into data/raw/ "
            "(see slice 1 / README)."
        )
    return pd.read_csv(csv_path, parse_dates=["date"])


def load_raw_rankings(path: Path | None = None) -> pd.DataFrame:
    """Load the FIFA ranking CSV.

    Args:
        path: Explicit path to the CSV. Defaults to ``data/raw/fifa_ranking.csv``.

    Returns:
        One row per team per ranking-release date (raw, unvalidated).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    csv_path = path or (RAW_DIR / RANKINGS_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the FIFA ranking dataset into data/raw/."
        )
    return pd.read_csv(csv_path, parse_dates=["rank_date"])
