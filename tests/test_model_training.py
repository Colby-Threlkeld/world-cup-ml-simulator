"""Tests for the main model training pipeline (train + calibrate + CLI).

Uses a small synthetic fixture with a genuine ``elo_diff`` signal so "the model
trains and produces sane probabilities" is a real result, not a faked one.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_model as cli  # noqa: E402
from worldcup.data.clean_data import TARGET_CLASSES  # noqa: E402
from worldcup.data.validate_data import DataValidationError  # noqa: E402
from worldcup.features.rolling_features import DIFFERENCE_FEATURES  # noqa: E402
from worldcup.models.calibrate import (  # noqa: E402
    expected_calibration_error,
    fit_calibrated_model,
)
from worldcup.models.train import (  # noqa: E402
    build_estimator,
    run_training,
    select_features,
)


def _make_features(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Feature frame with a real elo/fifa signal; form features are noise."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 150, n)
    logits = np.stack([elo_diff / 80.0, np.zeros(n), -elo_diff / 80.0], axis=1)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    classes = np.array(TARGET_CLASSES)
    y = np.array([rng.choice(classes, p=probs[i]) for i in range(n)])

    data = {
        "match_id": np.arange(n),
        "date": pd.date_range("1995-01-01", periods=n, freq="D"),
        "team_a": "A",
        "team_b": "B",
        "team_a_score": rng.integers(0, 4, n),
        "team_b_score": rng.integers(0, 4, n),
        "target_class": y,
        "elo_diff": elo_diff,
        "fifa_points_diff": elo_diff * 2 + rng.normal(0, 40, n),
        "is_team_a_home": rng.integers(0, 2, n).astype(bool),
        "is_neutral": rng.integers(0, 2, n).astype(bool),
    }
    for col in DIFFERENCE_FEATURES:
        data[col] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


@pytest.fixture()
def features() -> pd.DataFrame:
    return _make_features()


# --- unit-level -------------------------------------------------------------


def test_select_features_skips_missing_and_rejects_labels(features: pd.DataFrame) -> None:
    chosen = select_features(features, ["elo_diff", "not_a_column", "is_team_a_home"])
    assert chosen == ["elo_diff", "is_team_a_home"]
    with pytest.raises(DataValidationError):
        select_features(features, ["team_a_score"])  # a label must never be a feature


def test_build_estimator_types() -> None:
    assert build_estimator({"type": "logistic"}).steps[-1][0] == "clf"
    assert build_estimator({"type": "gradient_boosting"}).steps[-1][0] == "clf"
    with pytest.raises(DataValidationError):
        build_estimator({"type": "nonsense"})


def test_calibration_none_is_passthrough(features: pd.DataFrame) -> None:
    est = build_estimator({"type": "logistic"})
    X, y = features[["elo_diff"]], features["target_class"]
    est.fit(X, y)
    assert fit_calibrated_model(est, X, y, method="none") is est
    with pytest.raises(ValueError):
        fit_calibrated_model(est, X, y, method="bogus")


def test_expected_calibration_error_perfect_is_zero() -> None:
    y = np.array(["team_a_win", "draw", "team_b_win"])
    proba = np.eye(3)
    assert expected_calibration_error(y, proba, list(TARGET_CLASSES)) == pytest.approx(0.0)


# --- training: probabilities & temporal discipline --------------------------


def test_run_training_produces_valid_probabilities(features: pd.DataFrame) -> None:
    result = run_training(features, {"calibration": "sigmoid"})
    proba_cols = [f"p_{c}" for c in TARGET_CLASSES]
    probs = result.predictions[proba_cols].to_numpy()
    assert probs.shape == (len(result.predictions), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_can_train_on_tiny_fixture() -> None:
    result = run_training(_make_features(n=60), {"calibration": "sigmoid"})
    assert len(result.predictions) > 0
    assert result.features  # at least one feature was used


def test_test_split_occurs_after_train_split(features: pd.DataFrame) -> None:
    # The predictions are the test window; every test date must be strictly later
    # than the latest training date used (no random leakage across time).
    from worldcup.models.evaluate import temporal_train_val_test_split

    train, val, test = temporal_train_val_test_split(features)
    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()

    result = run_training(features, {"calibration": "sigmoid"})
    assert result.predictions["date"].min() >= train["date"].max()


def test_metrics_include_baseline_comparison(features: pd.DataFrame) -> None:
    result = run_training(features, {"calibration": "sigmoid"})
    cmp = result.metrics["comparison"]
    assert {"main_calibrated_test_log_loss", "best_baseline_test_log_loss", "beats_best_baseline"} <= set(cmp)
    assert "uniform_random" in result.metrics["baselines"]
    for split in ("validation", "test"):
        for variant in result.metrics["main_model"][split].values():
            assert {"log_loss", "brier", "accuracy"} <= set(variant)


def test_missing_label_column_raises() -> None:
    bad = _make_features(n=40).drop(columns=["target_class"])
    with pytest.raises(DataValidationError):
        run_training(bad)


# --- CLI: saved artifacts ---------------------------------------------------


def test_cli_writes_all_artifacts(tmp_path: Path) -> None:
    feats_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "model"
    _make_features().to_csv(feats_csv, index=False)

    rc = cli.main(
        ["--features", str(feats_csv), "--output-dir", str(out_dir), "--calibration", "sigmoid"]
    )
    assert rc == 0
    for fname in (cli.MODEL_FILE, cli.FEATURES_FILE, cli.METRICS_FILE, cli.PREDICTIONS_FILE):
        assert (out_dir / fname).exists(), f"missing artifact {fname}"

    # The saved model loads and predicts valid probabilities.
    model = joblib.load(out_dir / cli.MODEL_FILE)
    feature_list = json.loads((out_dir / cli.FEATURES_FILE).read_text())["features"]
    preds = pd.read_csv(feats_csv).head(5)
    proba = model.predict_proba(preds[feature_list])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    metrics = json.loads((out_dir / cli.METRICS_FILE).read_text())
    assert "comparison" in metrics


def test_cli_missing_features_file_returns_error(tmp_path: Path) -> None:
    assert cli.main(["--features", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path / "m")]) == 1


def test_cli_sample_mode_limits_training_rows(tmp_path: Path) -> None:
    feats_csv = tmp_path / "features.csv"
    out_dir = tmp_path / "model"
    _make_features(n=400).to_csv(feats_csv, index=False)

    rc = cli.main(
        ["--features", str(feats_csv), "--output-dir", str(out_dir), "--sample", "120", "--calibration", "sigmoid"]
    )
    assert rc == 0
    # 120 sampled rows, 15% test fraction -> ~18 predictions.
    preds = pd.read_csv(out_dir / cli.PREDICTIONS_FILE)
    assert 0 < len(preds) <= 30
