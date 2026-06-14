"""Shared pytest fixtures.

Exposes the committed sample dataset under ``tests/fixtures/`` so the suite can
exercise the real ingestion pipeline **without** the gitignored production data.
CI therefore needs no downloads — everything it touches is checked into the repo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the committed test-fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_results_path() -> Path:
    """Path to the tiny sample ``results.csv`` (martj42 schema)."""
    return FIXTURES_DIR / "sample_results.csv"


@pytest.fixture
def sample_raw_matches(sample_results_path: Path) -> pd.DataFrame:
    """The sample results loaded as a raw, unvalidated DataFrame."""
    return pd.read_csv(sample_results_path)
