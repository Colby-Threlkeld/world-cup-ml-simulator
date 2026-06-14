"""Tests for leakage-safe rating features (as-of Elo + FIFA joins).

The crux is the as-of boundary: a rating dated on or before the match is usable,
a rating dated even one day after must never be picked.
"""

import pandas as pd
import pytest

from worldcup.data.validate_data import LeakageError
from worldcup.features.build_features import build_feature_matrix
from worldcup.features.rating_features import (
    FIFA_FEATURES,
    RATING_FEATURES,
    add_rating_features,
)


def _model_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal model dataset from (match_id, date, team_a, team_b) rows."""
    return pd.DataFrame(
        {
            "match_id": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "team_a": [r[2] for r in rows],
            "team_b": [r[3] for r in rows],
        }
    )


def _elo(rows: list[tuple]) -> pd.DataFrame:
    """Build an Elo ratings frame from (team, date, elo) rows."""
    return pd.DataFrame(
        {
            "team": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "elo": [r[2] for r in rows],
        }
    )


def _fifa(rows: list[tuple]) -> pd.DataFrame:
    """Build a FIFA ranking frame from (team, rank_date, rank, points) rows."""
    return pd.DataFrame(
        {
            "team": [r[0] for r in rows],
            "rank_date": pd.to_datetime([r[1] for r in rows]),
            "rank": [r[2] for r in rows],
            "points": [r[3] for r in rows],
        }
    )


# --- the as-of boundary (the required date cases) ---------------------------


def test_rating_on_exact_match_date_is_used() -> None:
    df = add_rating_features(
        _model_df([(1, "2000-06-01", "A", "B")]),
        _elo([("A", "2000-06-01", 1500.0), ("B", "2000-06-01", 1400.0)]),
    ).set_index("match_id")
    assert df.loc[1, "team_a_elo"] == 1500.0
    assert df.loc[1, "team_b_elo"] == 1400.0
    assert df.loc[1, "elo_diff"] == pytest.approx(100.0)


def test_rating_one_day_before_match_is_used() -> None:
    df = add_rating_features(
        _model_df([(1, "2000-06-02", "A", "B")]),
        _elo([("A", "2000-06-01", 1500.0), ("B", "2000-06-01", 1400.0)]),
    ).set_index("match_id")
    assert df.loc[1, "team_a_elo"] == 1500.0


def test_rating_one_day_after_match_is_not_used() -> None:
    # The only rating is dated the day AFTER kickoff -> must be skipped (NaN).
    df = add_rating_features(
        _model_df([(1, "2000-06-01", "A", "B")]),
        _elo([("A", "2000-06-02", 9999.0), ("B", "2000-06-02", 9999.0)]),
    ).set_index("match_id")
    assert pd.isna(df.loc[1, "team_a_elo"])
    assert pd.isna(df.loc[1, "team_b_elo"])
    assert pd.isna(df.loc[1, "elo_diff"])  # diff is NaN if either side is NaN


def test_multiple_prior_ratings_picks_most_recent() -> None:
    df = add_rating_features(
        _model_df([(1, "2000-06-10", "A", "B")]),
        _elo(
            [
                ("A", "2000-01-01", 1400.0),
                ("A", "2000-05-01", 1500.0),  # latest <= match date -> chosen
                ("A", "2000-07-01", 1600.0),  # after match -> ignored
                ("B", "2000-04-01", 1300.0),
            ]
        ),
    ).set_index("match_id")
    assert df.loc[1, "team_a_elo"] == 1500.0
    assert df.loc[1, "team_b_elo"] == 1300.0


def test_missing_team_rating_is_nan() -> None:
    # Team B has no Elo row at all.
    df = add_rating_features(
        _model_df([(1, "2000-06-01", "A", "B")]),
        _elo([("A", "2000-05-01", 1500.0)]),
    ).set_index("match_id")
    assert df.loc[1, "team_a_elo"] == 1500.0
    assert pd.isna(df.loc[1, "team_b_elo"])
    assert pd.isna(df.loc[1, "elo_diff"])


# --- FIFA ranking join ------------------------------------------------------


def test_fifa_rank_and_points_join_as_of() -> None:
    df = add_rating_features(
        _model_df([(1, "2000-06-15", "A", "B")]),
        None,
        _fifa(
            [
                ("A", "2000-06-01", 5, 1700.0),
                ("A", "2000-07-01", 4, 1750.0),  # after match -> ignored
                ("B", "2000-06-01", 12, 1500.0),
            ]
        ),
    ).set_index("match_id")
    assert df.loc[1, "team_a_fifa_rank"] == 5
    assert df.loc[1, "team_b_fifa_rank"] == 12
    assert df.loc[1, "fifa_rank_diff"] == 5 - 12
    assert df.loc[1, "team_a_fifa_points"] == 1700.0
    assert df.loc[1, "fifa_points_diff"] == pytest.approx(1700.0 - 1500.0)
    for col in FIFA_FEATURES:
        assert col in df.columns


# --- name normalization -----------------------------------------------------


def test_ratings_team_names_are_normalized() -> None:
    # Ratings use a known alias ("USA"); the match uses the canonical name.
    df = add_rating_features(
        _model_df([(1, "2000-06-01", "United States", "B")]),
        _elo([("USA", "2000-05-01", 1620.0), ("B", "2000-05-01", 1400.0)]),
    ).set_index("match_id")
    assert df.loc[1, "team_a_elo"] == 1620.0


# --- leakage validation -----------------------------------------------------


def test_validation_catches_future_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the as-of guard to fail by feeding a frame whose join would only
    # produce a future-dated rating, while disabling the backward selection.
    import worldcup.features.rating_features as rf

    real_merge_asof = pd.merge_asof

    def _bad_join(left, right, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["direction"] = "forward"  # deliberately pick a FUTURE rating
        kwargs["allow_exact_matches"] = False
        return real_merge_asof(left, right, **kwargs)

    monkeypatch.setattr(rf.pd, "merge_asof", _bad_join)
    with pytest.raises(LeakageError):
        add_rating_features(
            _model_df([(1, "2000-06-01", "A", "B")]),
            _elo([("A", "2000-07-01", 1500.0), ("B", "2000-07-01", 1400.0)]),
        )


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


def test_build_feature_matrix_attaches_ratings() -> None:
    feats = build_feature_matrix(
        _matches(
            [
                (1, "2000-01-01", "A", "B", 2, 1),
                (2, "2000-02-01", "A", "B", 0, 0),
            ]
        ),
        elo_ratings=_elo([("A", "1999-12-01", 1550.0), ("B", "1999-12-01", 1450.0)]),
        fifa_rankings=_fifa([("A", "1999-12-01", 3, 1800.0), ("B", "1999-12-01", 8, 1600.0)]),
    )
    for col in RATING_FEATURES:
        assert col in feats.columns
    assert feats.set_index("match_id").loc[1, "elo_diff"] == pytest.approx(100.0)


def test_build_feature_matrix_without_ratings_unchanged() -> None:
    feats = build_feature_matrix(_matches([(1, "2000-01-01", "A", "B", 2, 1)]))
    for col in RATING_FEATURES:
        assert col not in feats.columns


def test_add_rating_features_does_not_mutate_input() -> None:
    model_df = _model_df([(1, "2000-06-01", "A", "B")])
    before = model_df.copy(deep=True)
    add_rating_features(model_df, _elo([("A", "2000-05-01", 1500.0)]))
    pd.testing.assert_frame_equal(model_df, before)
