"""Run the real ingestion pipeline on the committed fixture data (no real datasets).

This is the CI guarantee: the whole load -> clean -> validate path works on the
small CSV under ``tests/fixtures/``, so the suite never depends on the gitignored
production data.
"""

from pathlib import Path

import pandas as pd

from worldcup.data.clean_data import clean_matches
from worldcup.data.load_data import RAW_MATCH_COLUMNS, load_raw_matches
from worldcup.data.validate_data import validate_matches


def test_fixture_file_exists_and_has_expected_schema(sample_results_path: Path) -> None:
    assert sample_results_path.exists()
    df = pd.read_csv(sample_results_path)
    for col in RAW_MATCH_COLUMNS:
        assert col in df.columns


def test_load_and_clean_fixture_end_to_end(sample_results_path: Path) -> None:
    raw = load_raw_matches(sample_results_path)
    cleaned = clean_matches(raw)

    # Two unplayed 2026 fixtures (null scores) are split out; 11 played remain.
    assert len(raw) == 13
    assert len(cleaned) == 11
    validate_matches(cleaned)  # must not raise


def test_clean_fixture_normalizes_team_names(sample_results_path: Path) -> None:
    cleaned = clean_matches(load_raw_matches(sample_results_path))
    teams = set(cleaned["home_team"]) | set(cleaned["away_team"])
    assert "United States" in teams  # the "USA" alias was canonicalized
    assert "USA" not in teams


def test_clean_fixture_result_labels_match_scores(sample_results_path: Path) -> None:
    cleaned = clean_matches(load_raw_matches(sample_results_path))
    draws = cleaned[cleaned["home_score"] == cleaned["away_score"]]
    assert (draws["result"] == "D").all()
    assert (draws["draw"]).all()
