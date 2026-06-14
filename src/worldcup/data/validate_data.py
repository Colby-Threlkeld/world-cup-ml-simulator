"""Data validation layer.

Catches bad data before it reaches feature engineering or modeling. Validators
*accumulate* problems and report them together in one readable error, so you see
every issue at once instead of fixing them one re-run at a time.

Two families:
    * :func:`check_matches` / :func:`validate_matches` — the cleaned matches table.
    * forward-looking validators reused by later slices:
      :func:`validate_no_future_ratings` (leakage), :func:`validate_probabilities`
      (predictions sum to 1), and :func:`validate_probability_bounds` (simulation
      outputs stay within [0, 1]).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from worldcup.data.team_names import normalize_team_name


class DataValidationError(ValueError):
    """Raised when data fails one or more validation checks."""


class LeakageError(DataValidationError):
    """Raised when data would leak future information into a feature or label."""


# Required columns of the cleaned matches table.
REQUIRED_MATCH_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral",
    "result",
)

# Optional boolean label columns validated for consistency when present.
_OPTIONAL_LABEL_COLUMNS: tuple[str, ...] = ("home_win", "draw", "away_win")

# International football began in 1872 (Scotland v England, 1872-11-30).
MIN_REASONABLE_DATE = pd.Timestamp("1872-01-01")


def check_matches(
    df: pd.DataFrame,
    *,
    min_date: pd.Timestamp = MIN_REASONABLE_DATE,
    max_date: pd.Timestamp | None = None,
) -> list[str]:
    """Run every matches-table check and return a list of problem descriptions.

    Non-raising: an empty list means the table is valid. Use
    :func:`validate_matches` to raise instead.

    Args:
        df: Candidate cleaned matches table.
        min_date: Earliest plausible match date.
        max_date: Latest plausible match date (defaults to tomorrow, so freshly
            played matches are allowed but the future is not).

    Returns:
        A list of human-readable error strings (empty if valid).
    """
    errors: list[str] = []

    # 1. Required columns exist. Without them, later checks would raise KeyError.
    missing = [col for col in REQUIRED_MATCH_COLUMNS if col not in df.columns]
    if missing:
        return [f"missing required columns: {missing}"]

    # 9. No missing values in required columns.
    null_counts = df[list(REQUIRED_MATCH_COLUMNS)].isna().sum()
    nulls = {col: int(n) for col, n in null_counts.items() if n > 0}
    if nulls:
        errors.append(f"missing values in required columns: {nulls}")

    # 2 & 10. Dates parse and fall in a reasonable range.
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        errors.append(f"'date' is not datetime (dtype={df['date'].dtype})")
    else:
        upper = max_date or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
        dated = df["date"].dropna()
        out_of_range = int(((dated < min_date) | (dated > upper)).sum())
        if out_of_range:
            errors.append(f"{out_of_range} date(s) outside [{min_date.date()}, {upper.date()}]")

    # 3. No duplicate match_id values.
    dup_ids = int(df["match_id"].duplicated().sum())
    if dup_ids:
        errors.append(f"{dup_ids} duplicate match_id value(s)")

    # 4. Scores are non-negative integers.
    for col in ("home_score", "away_score"):
        scores = df[col].dropna()
        if not pd.api.types.is_integer_dtype(df[col]) and (scores % 1 != 0).any():
            errors.append(f"'{col}' has non-integer values")
        if (scores < 0).any():
            errors.append(f"'{col}' has negative values")

    # 5. Team names are normalized (normalization is idempotent on canonical names).
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna())
    un_normalized = sorted({t for t in teams if isinstance(t, str) and normalize_team_name(t) != t})
    if un_normalized:
        errors.append(f"{len(un_normalized)} un-normalized team name(s): {un_normalized[:10]}")

    # 6. home_team and away_team differ.
    same_team = int((df["home_team"] == df["away_team"]).sum())
    if same_team:
        errors.append(f"{same_team} row(s) where home_team == away_team")

    # 7. Neutral field is boolean.
    if df["neutral"].dtype != bool:
        errors.append(f"'neutral' is not boolean (dtype={df['neutral'].dtype})")

    # 8. Result labels agree with the scores.
    errors.extend(_check_result_labels(df))

    return errors


def validate_matches(df: pd.DataFrame, **kwargs: object) -> None:
    """Validate the cleaned matches table, raising on any failure.

    Args:
        df: Candidate cleaned matches table.
        **kwargs: Forwarded to :func:`check_matches` (``min_date``/``max_date``).

    Raises:
        DataValidationError: If any check fails (message lists every problem).
    """
    errors = check_matches(df, **kwargs)  # type: ignore[arg-type]
    if errors:
        raise DataValidationError(_format_errors("matches table", errors))


# --- forward-looking validators (used by later slices) ----------------------


def validate_no_future_ratings(
    df: pd.DataFrame,
    *,
    rating_date_col: str,
    match_date_col: str,
) -> None:
    """Ensure no rating is dated after the match it describes (leakage guard).

    Args:
        df: Frame joining ratings to matches.
        rating_date_col: Column holding the rating's effective date.
        match_date_col: Column holding the match (kickoff) date.

    Raises:
        DataValidationError: If a date column is missing.
        LeakageError: If any rating date is strictly after its match date.
    """
    for col in (rating_date_col, match_date_col):
        if col not in df.columns:
            raise DataValidationError(f"column '{col}' not found")
    rating_date = pd.to_datetime(df[rating_date_col])
    match_date = pd.to_datetime(df[match_date_col])
    future = rating_date > match_date
    n_future = int(future.sum())
    if n_future:
        sample = df.loc[future, [rating_date_col, match_date_col]].head(3).to_string(index=False)
        raise LeakageError(
            f"{n_future} rating(s) dated AFTER their match — this leaks the future "
            f"into a feature. Example rows:\n{sample}"
        )


def validate_rating_features(
    df: pd.DataFrame,
    rating_date_cols: Sequence[str],
    *,
    match_date_col: str = "date",
) -> None:
    """Ensure every joined rating snapshot is dated at or before its match.

    Convenience wrapper that runs :func:`validate_no_future_ratings` for each
    rating-date column produced by an as-of join (e.g. one per side per source),
    so a single call guards the whole rating-feature block against leakage.
    A ``NaT`` rating date (no prior rating available) is treated as missing, not
    as leakage.

    Args:
        df: Frame with the joined rating snapshot dates and the match date.
        rating_date_cols: The rating effective-date columns to check.
        match_date_col: Column holding the match (kickoff) date.

    Raises:
        DataValidationError: If a referenced column is missing.
        LeakageError: If any rating date is strictly after its match date.
    """
    for col in rating_date_cols:
        validate_no_future_ratings(df, rating_date_col=col, match_date_col=match_date_col)


def validate_probabilities(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    tolerance: float = 1e-6,
) -> None:
    """Validate a mutually-exclusive probability distribution (e.g. H/D/A).

    Checks each value is within ``[0, 1]`` and each row sums to 1.

    Args:
        df: Frame of predictions.
        columns: The probability columns that together form one distribution.
        tolerance: Allowed numerical slack for bounds and the row sum.

    Raises:
        DataValidationError: If columns are missing, contain NaN, fall outside
            ``[0, 1]``, or any row does not sum to 1.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"missing probability columns: {missing}")

    block = df[list(columns)]
    errors: list[str] = []
    if block.isna().to_numpy().any():
        errors.append("probability columns contain NaN")
    if ((block < -tolerance) | (block > 1 + tolerance)).to_numpy().any():
        errors.append("probability values outside [0, 1]")
    row_sums = block.sum(axis=1)
    bad_sum = (row_sums - 1.0).abs() > tolerance
    if bad_sum.any():
        examples = [round(float(x), 4) for x in row_sums[bad_sum].head(3)]
        errors.append(f"{int(bad_sum.sum())} row(s) do not sum to 1 (e.g. {examples})")

    if errors:
        raise DataValidationError(_format_errors("probabilities", errors))


def validate_probability_bounds(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1e-9,
) -> None:
    """Validate that probability columns stay within ``[lower, upper]``.

    Unlike :func:`validate_probabilities`, this imposes no row-sum constraint —
    suitable for simulation outputs (e.g. per-team round-advancement odds).

    Args:
        df: Frame of simulation results.
        columns: Probability columns to bound-check.
        lower: Lower bound (inclusive).
        upper: Upper bound (inclusive).
        tolerance: Numerical slack.

    Raises:
        DataValidationError: If columns are missing or any value is out of bounds.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    block = df[list(columns)]
    out_of_bounds = (block < lower - tolerance) | (block > upper + tolerance)
    n_bad = int(out_of_bounds.to_numpy().sum())
    if n_bad:
        raise DataValidationError(
            f"{n_bad} simulation probability value(s) outside [{lower}, {upper}]"
        )


# --- internal helpers -------------------------------------------------------


def _check_result_labels(df: pd.DataFrame) -> list[str]:
    """Check that ``result`` (and any boolean labels) agree with the scores."""
    errors: list[str] = []
    valid = df["home_score"].notna() & df["away_score"].notna()
    if not valid.any():
        return errors
    home = df.loc[valid, "home_score"]
    away = df.loc[valid, "away_score"]
    expected = np.select([home.values > away.values, home.values < away.values], ["H", "A"], "D")

    result = df.loc[valid, "result"].astype("string").to_numpy()
    n_mismatch = int((result != expected).sum())
    if n_mismatch:
        errors.append(f"{n_mismatch} row(s) where 'result' disagrees with the scores")

    conditions = {
        "home_win": home.values > away.values,
        "draw": home.values == away.values,
        "away_win": home.values < away.values,
    }
    for col in _OPTIONAL_LABEL_COLUMNS:
        if col in df.columns:
            got = df.loc[valid, col].astype(bool).to_numpy()
            n_wrong = int((got != conditions[col]).sum())
            if n_wrong:
                errors.append(f"{n_wrong} row(s) where '{col}' disagrees with the scores")
    return errors


def _format_errors(label: str, errors: Sequence[str]) -> str:
    """Render accumulated errors as a single readable, multi-line message."""
    header = f"{len(errors)} validation error(s) in {label}:"
    return "\n".join([header, *(f"  - {error}" for error in errors)])
