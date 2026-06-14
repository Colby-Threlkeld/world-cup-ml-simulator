"""Tests for the Team A vs Team B model-dataset transformation."""

from pathlib import Path

import pandas as pd
import pytest

from worldcup.data.clean_data import team_a_result_label, team_a_target_class
from worldcup.data.validate_data import DataValidationError
from worldcup.features.build_features import MODEL_DATASET_COLUMNS, build_model_dataset


def _matches() -> pd.DataFrame:
    """A small cleaned-matches frame: a home win, a neutral draw, an away win."""
    return pd.DataFrame(
        {
            "match_id": [10, 11, 12],
            "date": pd.to_datetime(["2018-06-14", "2022-11-21", "2014-06-20"]),
            "home_team": ["Russia", "Netherlands", "Iran"],
            "away_team": ["Saudi Arabia", "Turkey", "Spain"],
            "home_score": pd.array([5, 1, 0], dtype="int16"),
            "away_score": pd.array([0, 1, 2], dtype="int16"),
            "neutral": [False, True, False],
            "tournament": pd.Categorical(["FIFA World Cup", "Friendly", "FIFA World Cup"]),
            "result": pd.Categorical(["H", "D", "A"], categories=["H", "D", "A"]),
            "venue_country": ["Russia", "Qatar", "Brazil"],
        }
    )


@pytest.fixture
def model_ds() -> pd.DataFrame:
    return build_model_dataset(_matches())


def test_columns_and_no_duplication(model_ds: pd.DataFrame) -> None:
    assert list(model_ds.columns) == list(MODEL_DATASET_COLUMNS)
    assert len(model_ds) == 3  # one row per match — never duplicated


def test_team_ab_mapping(model_ds: pd.DataFrame) -> None:
    row = model_ds.iloc[0]
    assert row["team_a"] == "Russia"
    assert row["team_b"] == "Saudi Arabia"
    assert row["team_a_score"] == 5
    assert row["team_b_score"] == 0
    assert row["host_country"] == "Russia"


def test_targets_consistent_with_scores(model_ds: pd.DataFrame) -> None:
    assert list(model_ds["team_a_result"]) == ["win", "draw", "loss"]
    assert list(model_ds["target_class"]) == ["team_a_win", "draw", "team_b_win"]


def test_home_advantage_preserved(model_ds: pd.DataFrame) -> None:
    # team_a is the home side, so is_team_a_home is the negation of is_neutral.
    assert list(model_ds["is_neutral"]) == [False, True, False]
    assert list(model_ds["is_team_a_home"]) == [True, False, True]


def test_target_categories_are_fixed(model_ds: pd.DataFrame) -> None:
    assert list(model_ds["target_class"].cat.categories) == ["team_a_win", "draw", "team_b_win"]
    assert list(model_ds["team_a_result"].cat.categories) == ["win", "draw", "loss"]


def test_labels_match_scalar_helpers(model_ds: pd.DataFrame) -> None:
    for _, row in model_ds.iterrows():
        assert row["team_a_result"] == team_a_result_label(row["team_a_score"], row["team_b_score"])
        assert row["target_class"] == team_a_target_class(row["team_a_score"], row["team_b_score"])


def test_no_missing_targets(model_ds: pd.DataFrame) -> None:
    assert model_ds["target_class"].notna().all()
    assert model_ds["team_a_result"].notna().all()


def test_does_not_mutate_input() -> None:
    matches = _matches()
    before = matches.copy(deep=True)
    build_model_dataset(matches)
    pd.testing.assert_frame_equal(matches, before)


def test_missing_columns_raises() -> None:
    matches = _matches().drop(columns=["neutral"])
    with pytest.raises(DataValidationError):
        build_model_dataset(matches)


def test_scalar_helpers() -> None:
    assert team_a_result_label(2, 0) == "win"
    assert team_a_result_label(1, 1) == "draw"
    assert team_a_result_label(0, 3) == "loss"
    assert team_a_target_class(2, 0) == "team_a_win"
    assert team_a_target_class(1, 1) == "draw"
    assert team_a_target_class(0, 3) == "team_b_win"


# --- optional real-data smoke (skipped when matches.parquet is absent) -------

REAL_MATCHES = Path(__file__).resolve().parents[1] / "data" / "interim" / "matches.parquet"


@pytest.mark.skipif(not REAL_MATCHES.exists(), reason="matches.parquet not built")
def test_real_matches_smoke() -> None:
    matches = pd.read_parquet(REAL_MATCHES)
    ds = build_model_dataset(matches)
    assert len(ds) == len(matches)  # no duplication
    assert list(ds.columns) == list(MODEL_DATASET_COLUMNS)
    assert ds["target_class"].notna().all()
    assert (ds["is_team_a_home"] == ~ds["is_neutral"]).all()
