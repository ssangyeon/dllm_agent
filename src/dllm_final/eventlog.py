"""Append-only, per-task JSONL event logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from .types import TaskState


class TaskJSONLLogger:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def path_for(self, state: TaskState) -> Path:
        return self.directory / f"{state.safe_task_id()}.jsonl"

    def log(
        self,
        state: TaskState,
        event: str,
        *,
        latency_ms: float = 0.0,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": state.task_id,
            "event": event,
            "step": state.step_count,
            "latency_ms": round(max(0.0, float(latency_ms)), 3),
            "payload": dict(payload or {}),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path_for(state).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
