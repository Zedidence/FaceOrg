"""Simple background task system for long-running web operations."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from faceorganizer.logging_config import get_logger

log = get_logger("web.tasks")

# Completed/errored tasks are evicted after this many seconds.
_TASK_TTL_SECONDS = 3600  # 1 hour

# Background eviction runs every half-TTL so stale tasks don't accumulate
# even when the app is idle (no new tasks being created).
_EVICT_INTERVAL_SECONDS = _TASK_TTL_SECONDS // 2


@dataclass
class TaskStatus:
    """Status of a background task."""

    id: str
    type: str  # "scan", "cluster", "export"
    status: str = "pending"  # pending, running, done, cancelled, error
    message: str = ""
    progress: dict = field(default_factory=dict)
    result: Any = None
    finished_at: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)


_tasks: dict[str, TaskStatus] = {}
_lock = threading.Lock()


def _evict_stale() -> None:
    """Remove finished tasks older than _TASK_TTL_SECONDS. Caller must hold _lock."""
    now = time.monotonic()
    stale = [
        tid
        for tid, t in _tasks.items()
        if t.finished_at is not None and (now - t.finished_at) > _TASK_TTL_SECONDS
    ]
    for tid in stale:
        del _tasks[tid]


def _start_eviction_thread() -> None:
    """Start a daemon thread that periodically evicts stale completed tasks."""

    def _loop():
        while True:
            time.sleep(_EVICT_INTERVAL_SECONDS)
            with _lock:
                _evict_stale()

    t = threading.Thread(target=_loop, daemon=True, name="task-eviction")
    t.start()


_start_eviction_thread()


def create_task(task_type: str) -> TaskStatus:
    task = TaskStatus(id=uuid.uuid4().hex[:12], type=task_type)
    with _lock:
        _tasks[task.id] = task
    return task


def get_task(task_id: str) -> TaskStatus | None:
    with _lock:
        return _tasks.get(task_id)


def run_in_background(task: TaskStatus, fn, *args, **kwargs) -> None:
    """Run fn in a background thread, updating task status."""

    def _wrapper():
        task.status = "running"
        log.info("Task %s (%s) started", task.id, task.type)
        try:
            result = fn(*args, **kwargs)
            if result is not None:
                task.result = result
            task.status = "cancelled" if task.stop_event.is_set() else "done"
            log.info("Task %s (%s) finished: %s", task.id, task.type, task.status)
        except Exception as e:
            task.message = str(e)
            task.status = "error"
            log.exception("Task %s (%s) failed", task.id, task.type)
        finally:
            task.finished_at = time.monotonic()

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
