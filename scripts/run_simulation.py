"""CLI: run the Monte Carlo tournament simulation; save probabilities + summary.

Loads the (cached) tournament config and an optional cached strengths/ratings
artifact, builds a prediction function, runs the simulation, and writes a CSV of
per-team probabilities plus a JSON summary. The model is **never retrained** here.

Usage::

    python scripts/run_simulation.py --quick                  # 1,000 simulations
    python scripts/run_simulation.py --simulations 10000      # full run
    python scripts/run_simulation.py --strengths ratings.csv  # real forecast
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from worldcup.config import RANDOM_SEED, REPORTS_DIR  # noqa: E402
from worldcup.simulation.tournament import (  # noqa: E402
    load_tournament_config,
    simulate_tournament,
    strength_predict_fn,
    summarize_simulation,
    uniform_predict_fn,
)

logger = logging.getLogger("run_simulation")

QUICK_SIMULATIONS = 1_000
FULL_SIMULATIONS = 10_000


def _load_strengths(path: Path) -> dict[str, float]:
    """Load a cached ``team,strength`` CSV into a dict."""
    frame = pd.read_csv(path)
    columns = {c.lower(): c for c in frame.columns}
    if "team" not in columns or "strength" not in columns:
        raise SystemExit(
            f"{path} must have 'team' and 'strength' columns; found {list(frame.columns)}"
        )
    return dict(
        zip(
            frame[columns["team"]].astype(str),
            frame[columns["strength"]].astype(float),
            strict=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the simulation and write outputs. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run the Monte Carlo World Cup simulation.")
    parser.add_argument(
        "--simulations", "-n", type=int, default=FULL_SIMULATIONS,
        help=f"number of simulations (default {FULL_SIMULATIONS})",
    )
    parser.add_argument(
        "--quick", action="store_true", help=f"shortcut for {QUICK_SIMULATIONS} simulations"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument(
        "--strengths", type=Path, default=None,
        help="optional cached CSV (team,strength); without it, a uniform predictor is used",
    )
    parser.add_argument("--config", type=Path, default=None, help="tournament config YAML")
    parser.add_argument(
        "--output-dir", type=Path, default=REPORTS_DIR / "simulation", help="output directory"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    n_simulations = QUICK_SIMULATIONS if args.quick else args.simulations
    config = load_tournament_config(args.config)

    if args.strengths is not None:
        if not args.strengths.exists():
            logger.error("strengths file not found: %s", args.strengths)
            return 1
        predict = strength_predict_fn(_load_strengths(args.strengths))
        logger.info("Using strength-based predictor from %s", args.strengths)
    else:
        predict = uniform_predict_fn()
        logger.warning(
            "No --strengths given: using a UNIFORM predictor. With placeholder slots "
            "(draw_status=%s) results reflect bracket structure only, not a real forecast.",
            config.draw_status,
        )

    start = time.perf_counter()
    probabilities = simulate_tournament(
        config, predict, n_simulations=n_simulations, seed=args.seed
    )
    runtime = time.perf_counter() - start

    summary = summarize_simulation(
        probabilities,
        n_simulations=n_simulations,
        seed=args.seed,
        runtime_seconds=runtime,
        draw_status=config.draw_status,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tournament_probabilities.csv"
    json_path = args.output_dir / "tournament_summary.json"
    probabilities.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Ran %d simulations in %.2fs", n_simulations, runtime)
    logger.info("Wrote %s and %s", csv_path, json_path)
    logger.info(
        "Top title contenders: %s",
        ", ".join(
            f"{row['team']} {row['win_world_cup_probability']:.1%}"
            for row in summary["top_title_contenders"][:5]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
