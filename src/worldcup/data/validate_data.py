"""Schema and integrity validation for the match results table.

Validation is intentionally strict and fails loudly: bad data should never make
it far enough to corrupt features or leak into the model.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)


class DataValidationError(ValueError):
    """Raised when the match table fails a schema or integrity check."""


def validate_matches(df: pd.DataFrame) -> None:
    """Validate the match results table, raising on the first failure.

    Checks, in order:
        * all required columns are present,
        * no missing values in the required columns,
        * score columns are non-negative and integer-valued.

    Args:
        df: Candidate match table.

    Raises:
        DataValidationError: If any check fails.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        raise DataValidationError(f"Null values in required columns: {nulls.to_dict()}")

    for col in ("home_score", "away_score"):
        scores = df[col].dropna()
        if not (scores % 1 == 0).all():
            raise DataValidationError(f"Column '{col}' contains non-integer values.")
        if (scores < 0).any():
            raise DataValidationError(f"Column '{col}' contains negative values.")
