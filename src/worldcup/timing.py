"""Tiny runtime-logging helpers, shared by the pipeline CLIs.

Knowing how long each stage takes matters on a small, metered Azure VM: it tells
you when to reach for ``--quick``/``--sample`` and confirms the cached fast paths
are actually firing. :func:`log_runtime` wraps a block and logs its wall-clock
time; :func:`is_up_to_date` powers the "skip recompute when nothing changed"
caching in the build scripts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def log_runtime(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log the wall-clock runtime of the wrapped block.

    Args:
        logger: Where to emit the timing line.
        label: Human name for the stage (e.g. ``"build-features"``).

    Yields:
        Nothing; the block runs inside the timer. The time is logged even if the
        block raises (the exception still propagates).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s finished in %.2fs", label, time.perf_counter() - start)


def is_up_to_date(output: Path | str, inputs: Iterable[Path | str]) -> bool:
    """Return ``True`` if ``output`` exists and is newer than every input.

    Used to skip deterministic recomputation (e.g. the feature table) when nothing
    upstream has changed — the cheap way to avoid burning VM minutes. Missing
    inputs are ignored; if no inputs exist, an existing output counts as current.

    Args:
        output: The artifact that would be (re)written.
        inputs: Source files the output is derived from.

    Returns:
        Whether the output can be reused as-is.
    """
    out = Path(output)
    if not out.exists():
        return False
    out_mtime = out.stat().st_mtime
    return all(Path(src).stat().st_mtime <= out_mtime for src in inputs if Path(src).exists())
