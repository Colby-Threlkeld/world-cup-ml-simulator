"""Tests for the runtime-logging / freshness helpers in worldcup.timing."""

import logging
import time
from pathlib import Path

import pytest

from worldcup.timing import is_up_to_date, log_runtime


def test_log_runtime_emits_a_timing_line(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("timing_test")
    with caplog.at_level(logging.INFO), log_runtime(logger, "demo-stage"):
        pass
    assert any("demo-stage finished in" in m for m in caplog.messages)


def test_log_runtime_logs_even_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("timing_test")
    with caplog.at_level(logging.INFO), pytest.raises(ValueError), log_runtime(logger, "boom"):
        raise ValueError("kaboom")
    assert any("boom finished in" in m for m in caplog.messages)


def test_is_up_to_date_true_when_output_newer(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    out = tmp_path / "out.txt"
    src.write_text("x", encoding="utf-8")
    time.sleep(0.01)
    out.write_text("y", encoding="utf-8")
    assert is_up_to_date(out, [src]) is True


def test_is_up_to_date_false_when_input_newer(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    src = tmp_path / "in.txt"
    out.write_text("y", encoding="utf-8")
    time.sleep(0.01)
    src.write_text("x", encoding="utf-8")  # input touched after the output
    assert is_up_to_date(out, [src]) is False


def test_is_up_to_date_false_when_output_missing(tmp_path: Path) -> None:
    assert is_up_to_date(tmp_path / "nope.txt", [tmp_path]) is False


def test_is_up_to_date_ignores_missing_inputs(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    out.write_text("y", encoding="utf-8")
    # A non-existent input doesn't make the output stale.
    assert is_up_to_date(out, [tmp_path / "ghost.txt"]) is True
