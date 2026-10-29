"""
Utilities for measuring and recording execution times.
Separately records training and sample generation times.
"""

from __future__ import annotations

import time
import csv
from datetime import datetime, timezone
import fcntl
import os
import uuid
import warnings
from pathlib import Path
from typing import Any

from src.io_utils import load_json, save_json

GENERATION_TIMING_FILENAME = 'generation_timing.csv'
PHASE_COLUMNS = ('execution_id', 'phase_id', 'run_id', 'experiment_id', 'generator',
    'dataset', 'seed', 'class', 'requested_class', 'phase', 'status',
    'started_at_utc', 'finished_at_utc', 'elapsed_seconds', 'input_rows',
    'requested_per_class', 'returned_rows', 'cache_reused', 'pid', 'artifact_dir',
    'error_type', 'error_message')


def append_phase_record(path, record=None):
    """Append and fsync one event under an exclusive lock; never replace history.

    With no record, create/validate just the CSV header. Column changes require
    an explicit migration, not appending incompatible rows to an existing CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', newline='', encoding='utf-8') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            header = next(csv.reader(stream), None)
            if header is not None and header != list(PHASE_COLUMNS):
                raise ValueError(f'Incompatible generation timing CSV header: {path}')
            stream.seek(0, os.SEEK_END)
            writer = csv.DictWriter(stream, fieldnames=PHASE_COLUMNS)
            if header is None:
                writer.writeheader()
            if record is not None:
                writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PhaseTiming:
    """Two append-only events per call, linked by phase_id.

    Elapsed wall time uses perf_counter, excluding timing-CSV writes. `fit`
    covers the wrapper's complete fit/persistence, not just neural training;
    `sampling` includes restore, generation/inverse transforms or cache access.
    A hard kill leaves a started event with unknown duration, never a fake zero.
    This observer installs no signal handlers and starts no work itself.
    """
    def __init__(self, path, **context):
        unknown = set(context)-set(PHASE_COLUMNS)
        if unknown:
            raise ValueError(f'Unknown timing fields: {sorted(unknown)}')
        self.path = Path(path)
        self.record = {**context, 'phase_id': uuid.uuid4().hex, 'pid': os.getpid()}
        self.timer = Timer(context.get('phase', ''))

    def __enter__(self):
        self.record.update(status='started', started_at_utc=datetime.now(timezone.utc).isoformat())
        append_phase_record(self.path, self.record)
        self.timer.start()
        return self.record

    def __exit__(self, exc_type, exc, tb):
        self.timer.stop()
        self.record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=format(self.timer.elapsed, '.9f'),
                           status='completed' if exc_type is None else
                                  'failed' if isinstance(exc, Exception) else 'interrupted')
        if exc_type is not None:
            self.record.update(error_type=exc_type.__name__,
                               error_message=str(exc).replace('\r', '\\r').replace('\n', '\\n'))
        try:
            append_phase_record(self.path, self.record)
        except Exception as logging_error:
            if exc_type is None:
                raise
            warnings.warn(f'Could not persist terminal timing event: {logging_error}; original failure preserved',
                          RuntimeWarning)
        return False


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
