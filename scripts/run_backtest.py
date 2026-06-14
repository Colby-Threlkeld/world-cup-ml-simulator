"""CLI: backtest the match model against the 2014 / 2018 / 2022 World Cups.

Loads the cleaned matches, builds the leakage-safe feature matrix, adds a
walk-forward Elo, then for each tournament trains only on prior matches and scores
that tournament's 64 games. Writes a Markdown report plus a metrics JSON and a
per-tournament predictions CSV.

Usage::

    python scripts/run_backtest.py
    python scripts/run_backtest.py --matches data/interim/matches.parquet \
        --output-dir reports/backtesting --report reports/backtesting_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from worldcup.backtesting import (  # noqa: E402
    add_elo_features,
    backtest_report,
    render_backtest_markdown,
    run_backtests,
)
from worldcup.config import INTERIM_DIR, REPORTS_DIR  # noqa: E402
from worldcup.features.build_features import build_feature_matrix  # noqa: E402

logger = logging.getLogger("run_backtest")

DEFAULT_MATCHES = INTERIM_DIR / "matches.parquet"
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "backtesting"
DEFAULT_REPORT = REPORTS_DIR / "backtesting_report.md"


def main(argv: list[str] | None = None) -> int:
    """Run the backtest and write outputs. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.matches.exists():
        logger.error("matches file not found: %s (run scripts/build_matches.py first)", args.matches)
        return 1

    logger.info("Building features from %s", args.matches)
    matches = pd.read_parquet(args.matches)
    features = add_elo_features(build_feature_matrix(matches))

    results = run_backtests(features)
    if not results:
        logger.error("no tournaments were backtested")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = backtest_report(results)
    (args.output_dir / "backtest_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for result in results:
        slug = result.name.split()[0]  # e.g. "2014"
        result.predictions.to_csv(args.output_dir / f"predictions_{slug}.csv", index=False)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_backtest_markdown(results), encoding="utf-8")

    for row in report["tournaments"]:
        logger.info(
            "%s: log_loss=%.4f acc=%.3f | %s ranked %s/%s (top5=%s)",
            row["tournament"], row["log_loss"], row["accuracy"], row["champion"],
            row["champion_predicted_rank"], row["n_participants"], row["champion_in_top_5"],
        )
    logger.info("Wrote report -> %s and artifacts -> %s", args.report, args.output_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the model against past World Cups.")
    parser.add_argument(
        "--matches", type=Path, default=DEFAULT_MATCHES, help=f"cleaned matches (default: {DEFAULT_MATCHES})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"artifacts dir (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help=f"report path (default: {DEFAULT_REPORT})"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
