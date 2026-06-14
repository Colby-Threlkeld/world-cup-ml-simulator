"""Tests for leakage-safe rolling team features.

The headline test builds a fixture where a future match has a huge score and
verifies it cannot change any earlier match's features.
"""

from pathlib import Path

import pandas as pd
import pytest

from worldcup.features.build_features import build_feature_matrix
from worldcup.features.rolling_features import (
    DIFFERENCE_FEATURES,
    TEAM_FEATURES,
    add_rolling_features,
)


def _model_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal model dataset from (match_id, date, team_a, team_b, a, b) rows."""
    return pd.DataFrame(
        {
            "match_id": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "team_a": [r[2] for r in rows],
            "team_b": [r[3] for r in rows],
            "team_a_score": pd.array([r[4] for r in rows], dtype="int16"),
            "team_b_score": pd.array([r[5] for r in rows], dtype="int16"),
        }
    )


# --- the headline no-leakage test -------------------------------------------


def test_future_huge_score_does_not_affect_earlier_features() -> None:
    base = [
        (1, "2000-01-01", "A", "X", 1, 0),
        (2, "2000-02-01", "A", "Y", 2, 0),
        (3, "2000-03-01", "A", "Z", 3, 0),
    ]
    future = (4, "2000-04-01", "A", "W", 100, 0)  # a huge FUTURE score

    with_future = add_rolling_features(_model_df([*base, future]))
    without_future = add_rolling_features(_model_df(base))

    feature_cols = [
        c for c in with_future.columns if c.endswith(("_a", "_b")) or c in DIFFERENCE_FEATURES
    ]
    earlier_with = with_future[with_future["match_id"].isin([1, 2, 3])].set_index("match_id")
    earlier_without = without_future.set_index("match_id")

    # The future match (and its 100-goal score) must not change matches 1-3 at all.
    pd.testing.assert_frame_equal(
        earlier_with[feature_cols].sort_index(),
        earlier_without[feature_cols].sort_index(),
        check_like=True,
    )


def test_excludes_own_and_future_match_explicitly() -> None:
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "X", 1, 0),
                (2, "2000-02-01", "A", "Y", 2, 0),
                (3, "2000-03-01", "A", "Z", 3, 0),
                (4, "2000-04-01", "A", "W", 100, 0),
            ]
        )
    ).set_index("match_id")
    assert pd.isna(df.loc[1, "last_5_goals_for_avg_a"])  # no prior history
    assert df.loc[2, "last_5_goals_for_avg_a"] == 1.0  # only m1 (excludes own 2)
    assert df.loc[3, "last_5_goals_for_avg_a"] == 1.5  # m1,m2 (excludes own 3 and future 100)
    assert df.loc[4, "last_5_goals_for_avg_a"] == 2.0  # m1,m2,m3 (excludes own 100)


# --- individual features ----------------------------------------------------


def test_points_per_match() -> None:
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "X", 1, 0),  # win  -> 3
                (2, "2000-02-01", "A", "Y", 0, 0),  # draw -> 1
                (3, "2000-03-01", "A", "Z", 0, 2),  # loss -> 0
                (4, "2000-04-01", "A", "W", 5, 0),  # win  -> 3
            ]
        )
    ).set_index("match_id")
    assert df.loc[3, "last_5_points_per_match_a"] == pytest.approx((3 + 1) / 2)
    assert df.loc[4, "last_5_points_per_match_a"] == pytest.approx((3 + 1 + 0) / 3)


def test_days_since_last_match() -> None:
    df = add_rolling_features(
        _model_df([(1, "2000-01-01", "A", "X", 1, 0), (2, "2000-02-01", "A", "Y", 2, 0)])
    ).set_index("match_id")
    assert pd.isna(df.loc[1, "days_since_last_match_a"])  # first match
    assert df.loc[2, "days_since_last_match_a"] == 31


def test_matches_played_last_365_days() -> None:
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "X", 1, 0),
                (2, "2000-06-01", "A", "Y", 2, 0),
                (3, "2000-12-01", "A", "Z", 3, 0),
            ]
        )
    ).set_index("match_id")
    assert df.loc[1, "matches_played_last_365_days_a"] == 0
    assert df.loc[2, "matches_played_last_365_days_a"] == 1
    assert df.loc[3, "matches_played_last_365_days_a"] == 2


def test_matches_played_last_365_days_excludes_old() -> None:
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "X", 1, 0),
                (2, "2002-01-01", "A", "Y", 2, 0),  # 2 years later
            ]
        )
    ).set_index("match_id")
    assert df.loc[2, "matches_played_last_365_days_a"] == 0  # the old match drops out


def test_difference_features() -> None:
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "B", 3, 0),  # A beats B
                (2, "2000-02-01", "A", "B", 1, 0),  # A beats B again
                (3, "2000-03-01", "A", "B", 0, 0),  # inspect: both have 2 prior matches
            ]
        )
    ).set_index("match_id")
    row = df.loc[3]
    # A prior points avg = 3.0, B prior = 0.0
    assert row["form_5_diff"] == pytest.approx(3.0)
    # A prior goals_for avg = (3+1)/2 = 2.0, B = 0.0
    assert row["goals_for_5_diff"] == pytest.approx(2.0)
    # consistency: each diff equals the underlying a - b
    assert row["rest_days_diff"] == pytest.approx(
        row["days_since_last_match_a"] - row["days_since_last_match_b"]
    )


# --- edge cases -------------------------------------------------------------


def test_first_match_has_missing_rolling_features() -> None:
    df = add_rolling_features(_model_df([(1, "2000-01-01", "A", "B", 1, 0)])).set_index("match_id")
    for feature in TEAM_FEATURES:
        if feature == "matches_played_last_365_days":
            assert df.loc[1, f"{feature}_a"] == 0  # genuine count, not missing
        else:
            assert pd.isna(df.loc[1, f"{feature}_a"])


def test_limited_history_uses_available_matches() -> None:
    # With only 2 prior matches, a 5- or 10-window still averages what's available.
    df = add_rolling_features(
        _model_df(
            [
                (1, "2000-01-01", "A", "X", 2, 0),
                (2, "2000-02-01", "A", "Y", 4, 0),
                (3, "2000-03-01", "A", "Z", 0, 0),
            ]
        )
    ).set_index("match_id")
    assert df.loc[3, "last_10_goals_for_avg_a"] == pytest.approx((2 + 4) / 2)


def test_add_rolling_features_does_not_mutate_input() -> None:
    model_df = _model_df([(1, "2000-01-01", "A", "B", 1, 0), (2, "2000-02-01", "A", "B", 2, 0)])
    before = model_df.copy(deep=True)
    add_rolling_features(model_df)
    pd.testing.assert_frame_equal(model_df, before)


# --- integration via build_feature_matrix -----------------------------------


def _matches(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "home_team": [r[2] for r in rows],
            "away_team": [r[3] for r in rows],
            "home_score": pd.array([r[4] for r in rows], dtype="int16"),
            "away_score": pd.array([r[5] for r in rows], dtype="int16"),
            "neutral": [False] * len(rows),
            "tournament": pd.Categorical(["Friendly"] * len(rows)),
        }
    )


def test_build_feature_matrix_attaches_features() -> None:
    feats = build_feature_matrix(
        _matches(
            [
                (1, "2000-01-01", "A", "B", 2, 1),
                (2, "2000-02-01", "A", "B", 0, 0),
            ]
        )
    )
    assert len(feats) == 2
    for col in ("last_5_points_per_match_a", "last_5_points_per_match_b", *DIFFERENCE_FEATURES):
        assert col in feats.columns


REAL_MATCHES = Path(__file__).resolve().parents[1] / "data" / "interim" / "matches.parquet"


@pytest.mark.skipif(not REAL_MATCHES.exists(), reason="matches.parquet not built")
def test_build_feature_matrix_real_smoke() -> None:
    feats = build_feature_matrix(pd.read_parquet(REAL_MATCHES))
    matches = pd.read_parquet(REAL_MATCHES)
    assert len(feats) == len(matches)
    assert "form_5_diff" in feats.columns
    # The earliest match in history has no prior data for either side.
    earliest = feats.sort_values("date").iloc[0]
    assert pd.isna(earliest["last_5_goals_for_avg_a"])
    assert earliest["matches_played_last_365_days_a"] == 0
