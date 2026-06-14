"""Tests for the data validation layer (worldcup.data.validate_data)."""

import pandas as pd
import pytest

from worldcup.data.clean_data import add_result_label, match_result
from worldcup.data.validate_data import (
    DataValidationError,
    LeakageError,
    check_matches,
    validate_matches,
    validate_no_future_ratings,
    validate_probabilities,
    validate_probability_bounds,
)


def _valid_matches() -> pd.DataFrame:
    """A minimal, fully valid cleaned-matches frame."""
    return pd.DataFrame(
        {
            "match_id": [0, 1],
            "date": pd.to_datetime(["2018-06-14", "2018-06-15"]),
            "home_team": ["Russia", "Iran"],
            "away_team": ["Saudi Arabia", "Morocco"],
            "home_score": pd.array([5, 1], dtype="int16"),
            "away_score": pd.array([0, 0], dtype="int16"),
            "neutral": [False, True],
            "result": pd.Categorical(["H", "H"], categories=["H", "D", "A"]),
            "home_win": [True, True],
            "draw": [False, False],
            "away_win": [False, False],
        }
    )


# --- happy path -------------------------------------------------------------


def test_valid_matches_pass() -> None:
    assert check_matches(_valid_matches()) == []
    validate_matches(_valid_matches())  # should not raise


# --- each check fails clearly (checks 1-10) ---------------------------------


def test_missing_required_columns() -> None:
    df = _valid_matches().drop(columns=["result"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_matches(df)


def test_dates_not_parsed() -> None:
    df = _valid_matches()
    df["date"] = ["2018-06-14", "2018-06-15"]  # plain strings, not datetime
    assert any("'date' is not datetime" in e for e in check_matches(df))


def test_date_out_of_range() -> None:
    df = _valid_matches()
    # 1700 is before the 1872 floor but still inside datetime64[ns] range.
    df.loc[0, "date"] = pd.Timestamp("1700-01-01")
    assert any("outside" in e for e in check_matches(df))


def test_duplicate_match_id() -> None:
    df = _valid_matches()
    df["match_id"] = [7, 7]
    with pytest.raises(DataValidationError, match="duplicate match_id"):
        validate_matches(df)


def test_negative_score() -> None:
    df = _valid_matches()
    df["home_score"] = pd.array([-1, 1], dtype="int16")
    assert any("negative" in e for e in check_matches(df))


def test_non_integer_score() -> None:
    df = _valid_matches()
    df["home_score"] = [1.5, 1.0]
    assert any("non-integer" in e for e in check_matches(df))


def test_unnormalized_team_name() -> None:
    df = _valid_matches()
    df.loc[0, "home_team"] = "USA"  # should have been normalized to "United States"
    with pytest.raises(DataValidationError, match="un-normalized team name"):
        validate_matches(df)


def test_home_equals_away() -> None:
    df = _valid_matches()
    df.loc[0, "away_team"] = "Russia"
    with pytest.raises(DataValidationError, match="home_team == away_team"):
        validate_matches(df)


def test_neutral_not_boolean() -> None:
    df = _valid_matches()
    df["neutral"] = [0, 1]  # ints, not bools
    assert any("'neutral' is not boolean" in e for e in check_matches(df))


def test_result_disagrees_with_scores() -> None:
    df = _valid_matches()
    df["result"] = pd.Categorical(["A", "H"], categories=["H", "D", "A"])  # row 0 was a home win
    assert any("'result' disagrees" in e for e in check_matches(df))


def test_boolean_label_disagrees_with_scores() -> None:
    df = _valid_matches()
    df["home_win"] = [False, True]  # row 0 was a home win
    assert any("'home_win' disagrees" in e for e in check_matches(df))


def test_missing_required_value() -> None:
    df = _valid_matches()
    df.loc[0, "home_team"] = None
    with pytest.raises(DataValidationError, match="missing values"):
        validate_matches(df)


# --- aggregation & readability ----------------------------------------------


def test_multiple_errors_are_aggregated_and_readable() -> None:
    # Two *independent* problems (no cascading into result/label mismatches).
    df = _valid_matches()
    df["match_id"] = [3, 3]  # duplicate ids
    df["neutral"] = [0, 1]  # not boolean
    with pytest.raises(DataValidationError) as exc_info:
        validate_matches(df)
    message = str(exc_info.value)
    assert "2 validation error(s)" in message
    assert "duplicate match_id" in message
    assert "neutral" in message
    assert message.count("\n  - ") == 2  # one bullet per problem


# --- forward-looking validators ---------------------------------------------


def test_no_future_ratings_passes_when_as_of() -> None:
    df = pd.DataFrame(
        {
            "rating_date": pd.to_datetime(["2018-06-01", "2018-06-10"]),
            "match_date": pd.to_datetime(["2018-06-14", "2018-06-15"]),
        }
    )
    validate_no_future_ratings(df, rating_date_col="rating_date", match_date_col="match_date")


def test_future_rating_raises_leakage_error() -> None:
    df = pd.DataFrame(
        {
            "rating_date": pd.to_datetime(["2018-06-20"]),  # AFTER the match
            "match_date": pd.to_datetime(["2018-06-14"]),
        }
    )
    with pytest.raises(LeakageError):
        validate_no_future_ratings(df, rating_date_col="rating_date", match_date_col="match_date")


def test_probabilities_sum_to_one_passes() -> None:
    df = pd.DataFrame({"p_home": [0.5, 0.2], "p_draw": [0.3, 0.3], "p_away": [0.2, 0.5]})
    validate_probabilities(df, ["p_home", "p_draw", "p_away"])


def test_probabilities_not_summing_to_one_raise() -> None:
    df = pd.DataFrame({"p_home": [0.5], "p_draw": [0.2], "p_away": [0.1]})  # sums to 0.8
    with pytest.raises(DataValidationError, match="sum to 1"):
        validate_probabilities(df, ["p_home", "p_draw", "p_away"])


def test_probabilities_out_of_bounds_raise() -> None:
    df = pd.DataFrame({"p_home": [1.4], "p_draw": [-0.4], "p_away": [0.0]})
    with pytest.raises(DataValidationError, match=r"outside \[0, 1\]"):
        validate_probabilities(df, ["p_home", "p_draw", "p_away"])


def test_probability_bounds_pass_and_fail() -> None:
    ok = pd.DataFrame({"p_title": [0.1, 0.0, 1.0]})
    validate_probability_bounds(ok, ["p_title"])
    bad = pd.DataFrame({"p_title": [1.2]})
    with pytest.raises(DataValidationError, match="outside"):
        validate_probability_bounds(bad, ["p_title"])


# --- result-label helpers (clean_data) --------------------------------------


def test_match_result_and_add_label() -> None:
    assert match_result(2, 0) == "H"
    assert match_result(1, 1) == "D"
    assert match_result(0, 3) == "A"
    labeled = add_result_label(pd.DataFrame({"home_score": [2, 1], "away_score": [0, 3]}))
    assert list(labeled["result"]) == ["H", "A"]
