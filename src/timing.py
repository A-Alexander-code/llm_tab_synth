"""
Utilities for measuring and recording execution times.
Separately records training and sample generation times.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.io_utils import load_json, save_json


class Timer:
    """
    Context manager and accumulator for execution times.

    Usage as context manager:
        with Timer("training") as t:
            model.fit(...)
        print(t.elapsed)

    Cumulative usage:
        timer = Timer("sampling")
        timer.start()
        ...
        timer.stop()
        timer.start()
        ...
        timer.stop()
        print(timer.total)  # sum of all intervals
    """

    def __init__(self, label: str = ""):
        self.label    = label
        self.elapsed  = 0.0
        self.total    = 0.0
        self._start   = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start()")
        self.elapsed = time.perf_counter() - self._start
        self.total  += self.elapsed
        self._start  = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer(label={self.label!r}, total={self.total:.4f}s)"


def save_timing(
    timings: dict[str, float],
    path: Path,
) -> None:
    """
    Save a timing record to a JSON file.

    Args:
        timings: Dictionary with descriptive keys and values in seconds.
                 Example: {"training_class_0": 45.2, "sampling_class_0": 12.1}
        path: Output file path (timing.json)
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = load_json(path)
        except Exception:
            existing = {}

    existing.update(timings)
    save_json(existing, path)


def load_timing(path: Path) -> dict[str, float]:
    """
    Load a timing record from a JSON file.

    Returns:
        Dictionary with recorded times, or empty dict if not found.
    """
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}