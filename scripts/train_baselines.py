"""CLI: train and score the baseline match-outcome models.

Loads the feature table, makes a temporal train/validation/test split, fits every
baseline whose required features are present (skipping the rest with a warning),
scores each on validation and test with log loss / Brier / accuracy, and writes
the results to ``reports/baseline_metrics.json``.

These numbers are the bar the slice-4 model must clear. Only *played* matches feed
the baselines — the unplayed 2026 World Cup fixtures are dropped upstream by the
cleaning step and are never training data.

Usage::

    python scripts/train_baselines.py
    python scripts/train_baselines.py --features data/processed/features.csv \
        --output reports/baseline_metrics.json
    python scripts/train_baselines.py --sample 2000 -v   # quick smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from worldcup.data.clean_data import TARGET_CLASSES  # noqa: E402
from worldcup.data.validate_data import DataValidationError  # noqa: E402
from worldcup.models.baseline import available_baselines  # noqa: E402
from worldcup.models.evaluate import (  # noqa: E402
    evaluate_proba,
    save_metrics,
    temporal_train_val_test_split,
)

logger = logging.getLogger("train_baselines")

DEFAULT_FEATURES = PROCESSED_DIR / "features.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "baseline_metrics.json"
TARGET_COLUMN = "target_class"


def main(argv: list[str] | None = None) -> int:
    """Run the baseline training/evaluation pipeline. Returns an exit code."""
    t0 = time.perf_counter()
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        features = _load_features(args.features, sample=args.sample)
        train, val, test = temporal_train_val_test_split(
            features, val_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
        logger.info(
            "Temporal split: train=%d (%s..%s) val=%d test=%d",
            len(train), train["date"].min().date(), train["date"].max().date(),
            len(val), len(test),
        )

        metrics = _train_and_score(train, val, test)
        out_path = save_metrics(metrics, args.output)
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        logger.error("Baseline training failed: %s", exc)
        return 1

    _log_leaderboard(metrics)
    logger.info("Wrote baseline metrics for %d model(s) -> %s", len(metrics["models"]), out_path)
    logger.info("%s finished in %.2fs", "train-baselines", time.perf_counter() - t0)
    return 0


def _train_and_score(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Fit each available baseline and collect validation + test metrics."""
    y_train = train[TARGET_COLUMN]
    models = available_baselines(train)

    model_metrics: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(train, y_train)
        model_metrics[name] = {
            "validation": evaluate_proba(val[TARGET_COLUMN], model.predict_proba(val)),
            "test": evaluate_proba(test[TARGET_COLUMN], model.predict_proba(test)),
            "features": list(model.features),
        }
        logger.info(
            "  %-22s val log_loss=%.4f brier=%.4f acc=%.3f",
            name,
            model_metrics[name]["validation"]["log_loss"],
            model_metrics[name]["validation"]["brier"],
            model_metrics[name]["validation"]["accuracy"],
        )

    return {
        "split": {
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "train_end": str(train["date"].max().date()),
            "val_end": str(val["date"].max().date()),
            "test_end": str(test["date"].max().date()),
        },
        "classes": list(TARGET_CLASSES),
        "models": model_metrics,
    }


def _load_features(path: Path, *, sample: int | None) -> pd.DataFrame:
    """Load the feature CSV, keep only labeled (played) rows, optional head sample."""
    if not path.exists():
        raise FileNotFoundError(f"features file not found: {path}")
    features = pd.read_csv(path, parse_dates=["date"])
    for col in ("date", TARGET_COLUMN):
        if col not in features.columns:
            raise DataValidationError(f"features file missing required column '{col}'")

    # Defensive: only played matches carry a label. Unplayed 2026 fixtures (no
    # label) must never train a model — the cleaning step already drops them.
    labeled = features[features[TARGET_COLUMN].isin(TARGET_CLASSES)].copy()
    dropped = len(features) - len(labeled)
    if dropped:
        logger.warning("Dropped %d unlabeled (unplayed) row(s) before training", dropped)
    if labeled.empty:
        raise DataValidationError("no labeled matches available to train baselines")

    if sample is not None:
        labeled = labeled.sort_values("date", kind="stable").head(sample)
        logger.info("Sample mode: using the first %d labeled match(es)", len(labeled))
    return labeled


def _log_leaderboard(metrics: dict) -> None:
    """Log models ranked by test log loss (lower is better)."""
    ranked = sorted(metrics["models"].items(), key=lambda kv: kv[1]["test"]["log_loss"])
    logger.info("Test-set leaderboard (by log loss):")
    for name, m in ranked:
        t = m["test"]
        logger.info("  %-22s log_loss=%.4f brier=%.4f acc=%.3f", name, t["log_loss"], t["brier"], t["accuracy"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and score baseline models.")
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES, help=f"feature CSV (default: {DEFAULT_FEATURES})"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"metrics JSON (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="validation fraction (default: 0.15)")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="test fraction (default: 0.15)")
    parser.add_argument("--sample", type=int, default=None, help="use only the first N matches (quick smoke)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
