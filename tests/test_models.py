"""Tests for the baseline models, the evaluator, and the training script.

The fixture below builds a feature frame with a genuine signal (``elo_diff`` and
``fifa_points_diff`` predict the outcome), so "a logistic baseline beats uniform"
is a real result on held-out data — never a faked number.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_baselines as cli  # noqa: E402
from worldcup.data.clean_data import TARGET_CLASSES  # noqa: E402
from worldcup.features.rolling_features import DIFFERENCE_FEATURES  # noqa: E402
from worldcup.models.baseline import (  # noqa: E402
    ClassFrequencyBaseline,
    EloLogisticBaseline,
    FeatureUnavailableError,
    FifaPointsLogisticBaseline,
    RecentFormLogisticBaseline,
    UniformBaseline,
    WeightedEnsembleBaseline,
    available_baselines,
)
from worldcup.models.evaluate import (  # noqa: E402
    evaluate_proba,
    save_metrics,
    temporal_train_val_test_split,
)


def _make_features(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Feature frame where elo/fifa carry real signal and form is pure noise."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 150, n)
    # Latent class logits: stronger team_a (positive elo_diff) -> team_a_win.
    logits = np.stack([elo_diff / 80.0, np.zeros(n), -elo_diff / 80.0], axis=1)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    classes = np.array(TARGET_CLASSES)
    y = np.array([rng.choice(classes, p=probs[i]) for i in range(n)])

    data = {
        "match_id": np.arange(n),
        "date": pd.date_range("2000-01-01", periods=n, freq="D"),
        "team_a": "A",
        "team_b": "B",
        "target_class": y,
        "elo_diff": elo_diff,
        "fifa_points_diff": elo_diff * 2 + rng.normal(0, 40, n),
    }
    for col in DIFFERENCE_FEATURES:
        data[col] = rng.normal(0, 1, n)  # noise: form baseline should ~ match the prior
    return pd.DataFrame(data)


@pytest.fixture()
def features() -> pd.DataFrame:
    return _make_features()


# --- the shared interface contract ------------------------------------------

ALL_MODEL_FACTORIES = [
    UniformBaseline,
    ClassFrequencyBaseline,
    EloLogisticBaseline,
    FifaPointsLogisticBaseline,
    RecentFormLogisticBaseline,
]


@pytest.mark.parametrize("factory", ALL_MODEL_FACTORIES)
def test_predict_proba_shape_and_sums_to_one(factory, features: pd.DataFrame) -> None:
    model = factory().fit(features, features["target_class"])
    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    assert (proba > 0).all()  # no exact zeros -> log loss stays finite


def test_fit_returns_self(features: pd.DataFrame) -> None:
    model = EloLogisticBaseline()
    assert model.fit(features, features["target_class"]) is model


# --- individual baselines ---------------------------------------------------


def test_uniform_is_flat(features: pd.DataFrame) -> None:
    proba = UniformBaseline().fit(features, features["target_class"]).predict_proba(features)
    np.testing.assert_allclose(proba, 1.0 / 3.0)


def test_class_frequency_matches_smoothed_priors(features: pd.DataFrame) -> None:
    model = ClassFrequencyBaseline(alpha=1.0).fit(features, features["target_class"])
    proba = model.predict_proba(features)
    counts = features["target_class"].value_counts()
    expected = np.array([counts.get(c, 0) + 1.0 for c in TARGET_CLASSES])
    expected /= expected.sum()
    np.testing.assert_allclose(proba[0], expected)
    # Every row is identical (a prior, not a per-row prediction).
    np.testing.assert_allclose(proba, np.tile(proba[0], (len(features), 1)))


def test_elo_logistic_beats_uniform_on_heldout(features: pd.DataFrame) -> None:
    train, _, test = temporal_train_val_test_split(features)
    elo = EloLogisticBaseline().fit(train, train["target_class"])
    uniform = UniformBaseline().fit(train, train["target_class"])
    elo_ll = evaluate_proba(test["target_class"], elo.predict_proba(test))["log_loss"]
    uniform_ll = evaluate_proba(test["target_class"], uniform.predict_proba(test))["log_loss"]
    assert elo_ll < uniform_ll  # the real signal must pay off


def test_missing_feature_raises(features: pd.DataFrame) -> None:
    no_elo = features.drop(columns=["elo_diff"])
    with pytest.raises(FeatureUnavailableError):
        EloLogisticBaseline().fit(no_elo, no_elo["target_class"])


def test_ensemble_sums_to_one_and_blends(features: pd.DataFrame) -> None:
    ensemble = WeightedEnsembleBaseline(
        [ClassFrequencyBaseline(), EloLogisticBaseline()]
    ).fit(features, features["target_class"])
    proba = ensemble.predict_proba(features)
    assert proba.shape == (len(features), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_ensemble_rejects_empty() -> None:
    with pytest.raises(ValueError):
        WeightedEnsembleBaseline([])


# --- available_baselines (skip-with-warning) --------------------------------


def test_available_baselines_full_set(features: pd.DataFrame) -> None:
    models = available_baselines(features)
    assert set(models) == {
        "uniform_random",
        "class_frequency",
        "elo_logistic",
        "fifa_points_logistic",
        "recent_form_logistic",
        "weighted_ensemble",
    }


def test_available_baselines_skips_missing_features(
    features: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    only_prior = features.drop(columns=["elo_diff", "fifa_points_diff", *DIFFERENCE_FEATURES])
    with caplog.at_level("WARNING"):
        models = available_baselines(only_prior)
    assert set(models) == {"uniform_random", "class_frequency"}
    # No feature baselines -> ensemble is skipped too, with a warning.
    assert "weighted_ensemble" in " ".join(caplog.messages)


# --- evaluator --------------------------------------------------------------


def test_evaluate_proba_perfect_prediction() -> None:
    y = np.array(["team_a_win", "draw", "team_b_win"])
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Clip so log loss is finite, as the models do.
    proba = np.clip(proba, 1e-12, None)
    proba /= proba.sum(axis=1, keepdims=True)
    m = evaluate_proba(y, proba)
    assert m["accuracy"] == 1.0
    assert m["brier"] == pytest.approx(0.0, abs=1e-6)
    assert m["log_loss"] == pytest.approx(0.0, abs=1e-6)
    assert m["n"] == 3


def test_evaluate_proba_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        evaluate_proba(np.array(["draw"]), np.array([[0.5, 0.5]]))


def test_temporal_split_is_ordered_by_date(features: pd.DataFrame) -> None:
    train, val, test = temporal_train_val_test_split(features)
    assert train["date"].max() <= val["date"].min()
    assert val["date"].max() <= test["date"].min()
    assert len(train) + len(val) + len(test) == len(features)


def test_save_metrics_roundtrip(tmp_path: Path) -> None:
    path = save_metrics({"models": {"uniform_random": {"test": {"log_loss": 1.0}}}}, tmp_path / "m.json")
    assert json.loads(path.read_text(encoding="utf-8"))["models"]["uniform_random"]["test"]["log_loss"] == 1.0


# --- end-to-end through the training script ----------------------------------


def test_train_baselines_cli_writes_metrics(tmp_path: Path) -> None:
    feats_csv = tmp_path / "features.csv"
    out_json = tmp_path / "baseline_metrics.json"
    _make_features().to_csv(feats_csv, index=False)

    rc = cli.main(["--features", str(feats_csv), "--output", str(out_json)])
    assert rc == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(report["models"]) >= {"uniform_random", "elo_logistic", "weighted_ensemble"}
    for entry in report["models"].values():
        for split in ("validation", "test"):
            assert set(entry[split]) >= {"log_loss", "brier", "accuracy", "n"}


def test_train_baselines_cli_skips_when_no_features(tmp_path: Path) -> None:
    feats = _make_features().drop(columns=["elo_diff", "fifa_points_diff", *DIFFERENCE_FEATURES])
    feats_csv = tmp_path / "features.csv"
    out_json = tmp_path / "metrics.json"
    feats.to_csv(feats_csv, index=False)

    rc = cli.main(["--features", str(feats_csv), "--output", str(out_json)])
    assert rc == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(report["models"]) == {"uniform_random", "class_frequency"}


def test_train_baselines_cli_missing_file() -> None:
    assert cli.main(["--features", "definitely/not/here.csv"]) == 1
