"""State and observation types shared by every backend and tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, ClassVar, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .actions import Action


Status = Literal["running", "completed", "terminated"]


@dataclass(frozen=True)
class Observation:
    """A normalized tool result.

    ``raw_reference`` identifies the source or expression used. Every text field
    is capped here so custom tools cannot amplify prompts, logs, or CLI output.
    """

    summary: str
    raw_reference: str | None = None
    error: str | None = None
    reliability: float = 0.0

    MAX_SUMMARY_CHARS: ClassVar[int] = 4_000
    MAX_REFERENCE_CHARS: ClassVar[int] = 8_000
    MAX_ERROR_CHARS: ClassVar[int] = 2_000
    _TRUNCATION_MARKER: ClassVar[str] = "…[truncated]"

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("observation summary must be a non-empty string")
        if self.raw_reference is not None and not isinstance(self.raw_reference, str):
            raise TypeError("raw_reference must be a string or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if isinstance(self.reliability, bool) or not isinstance(self.reliability, (int, float)):
            raise TypeError("reliability must be numeric")
        if not 0.0 <= float(self.reliability) <= 1.0:
            raise ValueError("reliability must be between 0 and 1")
        object.__setattr__(self, "summary", self._cap(self.summary, self.MAX_SUMMARY_CHARS))
        if self.raw_reference is not None:
            object.__setattr__(
                self,
                "raw_reference",
                self._cap(self.raw_reference, self.MAX_REFERENCE_CHARS),
            )
        if self.error is not None:
            object.__setattr__(self, "error", self._cap(self.error, self.MAX_ERROR_CHARS))
        object.__setattr__(self, "reliability", float(self.reliability))

    @classmethod
    def _cap(cls, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        keep = limit - len(cls._TRUNCATION_MARKER)
        return value[:keep] + cls._TRUNCATION_MARKER

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StepRecord:
    step: int
    action_hash: str
    action: dict[str, Any]
    observation: Observation
    cached: bool
    progress: bool
    latency_ms: float


@dataclass
class TaskState:
    """Mutable state with all transitions concentrated in explicit methods."""

    task_id: str
    request: str
    status: Status = "running"
    answer: str | None = None
    termination_reason: str | None = None
    step_count: int = 0
    no_progress_count: int = 0
    model_calls: int = 0
    valid_actions: int = 0
    repair_attempts: int = 0
    repair_successes: int = 0
    observation_prompt_calls: int = 0
    duplicate_actions: int = 0
    total_latency_ms: float = 0.0
    history: list[StepRecord] = field(default_factory=list)
    seen_action_hashes: set[str] = field(default_factory=set)
    _observation_fingerprints: set[str] = field(default_factory=set, repr=False)

    MAX_TASK_ID_CHARS: ClassVar[int] = 256
    MAX_REQUEST_CHARS: ClassVar[int] = 32_000

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str):
            raise ValueError("task_id must be a string")
        if not isinstance(self.request, str):
            raise ValueError("request must be a string")
        if len(self.task_id) > self.MAX_TASK_ID_CHARS:
            raise ValueError(f"task_id exceeds {self.MAX_TASK_ID_CHARS} characters")
        if len(self.request) > self.MAX_REQUEST_CHARS:
            raise ValueError(f"request exceeds {self.MAX_REQUEST_CHARS} characters")
        if not self.task_id or self.task_id.isspace():
            raise ValueError("task_id must be a non-empty string")
        if not self.request or self.request.isspace():
            raise ValueError("request must be a non-empty string")

    def record_model_latency(self, latency_ms: float) -> None:
        self.total_latency_ms += max(0.0, float(latency_ms))

    def record_model_call(self, *, includes_observation: bool = False) -> None:
        self.model_calls += 1
        if includes_observation:
            self.observation_prompt_calls += 1

    def record_valid_action(self, *, repaired: bool = False) -> None:
        """Record a model response accepted by the exact action schema."""

        self.valid_actions += 1
        if repaired:
            self.repair_successes += 1

    def record_repair(self) -> None:
        self.repair_attempts += 1

    def apply_tool_result(
        self,
        action: "Action",
        observation: Observation,
        *,
        action_hash: str,
        cached: bool,
        latency_ms: float,
    ) -> bool:
        """Apply one tool transition and return whether it made new progress."""

        self._require_running()
        duplicate = action_hash in self.seen_action_hashes
        fingerprint = observation.fingerprint()
        # A non-cacheable polling tool may repeat an action yet return genuinely
        # new state (pending -> ready). Progress follows the observation, while
        # repeated actions remain separately visible in duplicate metrics.
        progress = not cached and fingerprint not in self._observation_fingerprints

        self.step_count += 1
        self.total_latency_ms += max(0.0, float(latency_ms))
        if duplicate or cached:
            self.duplicate_actions += 1
        self.seen_action_hashes.add(action_hash)
        self._observation_fingerprints.add(fingerprint)
        self.no_progress_count = 0 if progress else self.no_progress_count + 1
        self.history.append(
            StepRecord(
                step=self.step_count,
                action_hash=action_hash,
                action=action.to_dict(),
                observation=observation,
                cached=bool(cached),
                progress=progress,
                latency_ms=max(0.0, float(latency_ms)),
            )
        )
        return progress

    def apply_final(self, action: "Action") -> None:
        self._require_running()
        if action.kind != "final" or action.answer is None:
            raise ValueError("apply_final requires a final action")
        self.step_count += 1
        self.answer = action.answer
        self.status = "completed"
        self.termination_reason = "model_final"

    def terminate(self, reason: str, *, answer: str | None = None) -> None:
        self._require_running()
        self.status = "terminated"
        self.termination_reason = reason
        if answer is not None:
            self.answer = answer

    def to_prompt_snapshot(self, *, max_observations: int = 6) -> str:
        count = max(0, max_observations)
        recent = [] if count == 0 else self.history[-count:]
        data = {
            "task_id": self.task_id,
            "request": self.request,
            "step_count": self.step_count,
            "no_progress_count": self.no_progress_count,
            "observations": [
                {
                    "step": record.step,
                    "tool": record.action.get("tool"),
                    "summary": record.observation.summary,
                    "raw_reference": record.observation.raw_reference,
                    "error": record.observation.error,
                    "reliability": record.observation.reliability,
                    "cached": record.cached,
                }
                for record in recent
            ],
        }
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "request": self.request,
            "status": self.status,
            "answer": self.answer,
            "termination_reason": self.termination_reason,
            "step_count": self.step_count,
            "no_progress_count": self.no_progress_count,
            "model_calls": self.model_calls,
            "valid_actions": self.valid_actions,
            "repair_attempts": self.repair_attempts,
            "repair_successes": self.repair_successes,
            "observation_prompt_calls": self.observation_prompt_calls,
            "duplicate_actions": self.duplicate_actions,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "history": [
                {
                    **asdict(record),
                    "observation": asdict(record.observation),
                }
                for record in self.history
            ],
        }

    def safe_task_id(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_id).strip("._")[:64]
        digest = hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:10]
        return f"{slug or 'task'}-{digest}"

    def _require_running(self) -> None:
        if self.status != "running":
            raise RuntimeError(f"task is not running: {self.status}")
