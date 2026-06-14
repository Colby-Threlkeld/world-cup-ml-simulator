"""CLI: build the model-ready feature table from cleaned matches (+ ratings).

Loads the cleaned matches, reshapes them to the Team A vs Team B model dataset,
attaches leakage-safe rolling form features, and — when the optional Elo / FIFA
rating files are present — joins their as-of rating features on top. The result
is validated and written to ``data/processed/features.csv``.

The optional rating files are *optional*: if they are missing the script logs a
clear warning and continues with the rolling features only (it never crashes).

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --matches data/processed/matches.csv \
        --elo data/processed/elo_ratings.csv --fifa data/processed/fifa_rankings.csv \
        --output data/processed/features.csv
    python scripts/build_features.py --sample 500        # quick smoke on first 500 matches
    python scripts/build_features.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.config import PROCESSED_DIR  # noqa: E402
from worldcup.data.validate_data import DataValidationError  # noqa: E402
from worldcup.features.build_features import (  # noqa: E402
    DEFAULT_FEATURES_PATH,
    build_feature_matrix,
    load_matches,
    load_optional_ratings,
    missing_value_summary,
    save_features,
    validate_feature_matrix,
)

logger = logging.getLogger("build_features")

DEFAULT_MATCHES = PROCESSED_DIR / "matches.csv"
DEFAULT_ELO = PROCESSED_DIR / "elo_ratings.csv"
DEFAULT_FIFA = PROCESSED_DIR / "fifa_rankings.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the feature-building pipeline. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        matches = load_matches(args.matches)
        if args.sample is not None:
            # Quick smoke mode: the earliest N matches, so history stays coherent.
            matches = matches.sort_values("date", kind="stable").head(args.sample)
            logger.info("Sample mode: using the first %d match(es)", len(matches))
        logger.info("Loaded %d cleaned match(es) from %s", len(matches), args.matches)

        elo = load_optional_ratings(args.elo, date_col="date", label="Elo ratings")
        fifa = load_optional_ratings(args.fifa, date_col="rank_date", label="FIFA rankings")

        features = build_feature_matrix(matches, elo_ratings=elo, fifa_rankings=fifa)
        validate_feature_matrix(features, expected_rows=len(matches))
        _log_summary(features)
        out_path = save_features(features, args.output)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("Feature build failed: %s", exc)
        return 1

    logger.info("Done: %d feature rows x %d cols -> %s", *features.shape, out_path)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the model-ready feature table.")
    parser.add_argument(
        "--matches", type=Path, default=DEFAULT_MATCHES, help=f"cleaned matches (default: {DEFAULT_MATCHES})"
    )
    parser.add_argument(
        "--elo", type=Path, default=DEFAULT_ELO, help=f"optional Elo ratings CSV (default: {DEFAULT_ELO})"
    )
    parser.add_argument(
        "--fifa", type=Path, default=DEFAULT_FIFA, help=f"optional FIFA rankings CSV (default: {DEFAULT_FIFA})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_FEATURES_PATH, help=f"output CSV (default: {DEFAULT_FEATURES_PATH})"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="build features on only the first N matches (quick smoke)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def _log_summary(features: pd.DataFrame) -> None:
    """Log row count and the columns that carry missing values."""
    logger.info("Assembled %d feature rows x %d columns", *features.shape)
    missing = missing_value_summary(features)
    if missing.empty:
        logger.info("No missing values in any feature column.")
    else:
        top = ", ".join(f"{col}={n}" for col, n in missing.head(10).items())
        logger.info("Columns with missing values (top 10): %s", top)


if __name__ == "__main__":
    raise SystemExit(main())
