"""
job_store.py — Job state manager with disk persistence.
Thread-safe via a threading.Lock since BackgroundTasks run in a thread pool.
Jobs are persisted to temp/jobs.json for recovery across server restarts.
"""

import threading
import json
import pathlib
from typing import Any, Dict, Optional

# The central store: { job_id: { status, progress_percent, result, error } }
_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_store_file = pathlib.Path("temp/jobs.json")


def _load_store() -> None:
    """Load persisted jobs from disk."""
    global _store
    if _store_file.exists():
        try:
            with open(_store_file, "r") as f:
                _store = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load job store: {e}")


def _save_store() -> None:
    """Persist jobs to disk."""
    try:
        _store_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_store_file, "w") as f:
            json.dump(_store, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save job store: {e}")


# Load persisted jobs on module import
_load_store()


def create_job(job_id: str) -> None:
    """Initialise a new job entry with default state."""
    with _lock:
        _store[job_id] = {
            "status": "queued",
            "progress_percent": 0,
            "result": None,
            "error": None,
        }
        _save_store()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job dict or None if it doesn't exist."""
    with _lock:
        return dict(_store[job_id]) if job_id in _store else None


def update_job(job_id: str, **kwargs) -> None:
    """
    Update one or more fields of an existing job.
    Silently ignores updates for unknown job_ids.
    """
    with _lock:
        if job_id in _store:
            _store[job_id].update(kwargs)
            _save_store()


def all_jobs() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of the entire store (for debugging)."""
    with _lock:
        return {k: dict(v) for k, v in _store.items()}
