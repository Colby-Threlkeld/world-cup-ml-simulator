"""CLI: build the cleaned ``matches`` table from the raw results CSV.

Loads raw results, cleans/validates them, and writes the parquet table.

Usage::

    python scripts/build_matches.py
    python scripts/build_matches.py --input data/raw/results.csv --output data/interim/matches.parquet
    python scripts/build_matches.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.data.clean_data import (  # noqa: E402
    DEFAULT_MATCHES_PATH,
    clean_matches,
    save_matches,
)
from worldcup.data.load_data import load_raw_matches  # noqa: E402
from worldcup.data.validate_data import DataValidationError  # noqa: E402

logger = logging.getLogger("build_matches")


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Build the cleaned matches table.")
    parser.add_argument(
        "--input", type=Path, default=None, help="raw results CSV (default: data/raw/results.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help=f"output parquet (default: {DEFAULT_MATCHES_PATH})"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        raw = load_raw_matches(args.input)
        cleaned = clean_matches(raw)
        out_path = save_matches(cleaned, args.output)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    logger.info("Done: %d matches -> %s", len(cleaned), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
