"""Replayable benchmark harness and aggregate operational metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import fmean, median
from typing import Any, Callable, Iterable

from .adapters import ModelAdapter, ReplayAdapter
from .engine import AgentEngine
from .types import TaskState


@dataclass(frozen=True)
class BenchmarkCase:
    task_id: str
    request: str
    outputs: tuple[str, ...]
    expected_contains: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BenchmarkCase":
        required = {"task_id", "request", "outputs"}
        missing = required - set(value)
        extra = set(value) - required - {"expected_contains"}
        if missing or extra:
            raise ValueError(f"benchmark case keys: missing={sorted(missing)}, extra={sorted(extra)}")
        outputs = value["outputs"]
        if not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs):
            raise TypeError("benchmark outputs must be a list of strings")
        expected = value.get("expected_contains")
        if expected is not None and not isinstance(expected, str):
            raise TypeError("expected_contains must be a string or null")
        return cls(
            task_id=str(value["task_id"]),
            request=str(value["request"]),
            outputs=tuple(outputs),
            expected_contains=expected,
        )


@dataclass(frozen=True)
class BenchmarkMetrics:
    tasks: int
    completed: int
    completion_rate: float
    expected_matches: int
    expected_match_rate: float
    terminated: int
    average_steps: float
    average_latency_ms: float
    repair_task_rate: float
    duplicate_action_rate: float
    tool_error_rate: float

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in data.items()
        }


class BenchmarkRunner:
    """Run each case in a fresh engine supplied by ``engine_factory``."""

    def __init__(self, engine_factory: Callable[[ReplayAdapter], AgentEngine]) -> None:
        self.engine_factory = engine_factory

    def run(self, cases: Iterable[BenchmarkCase]) -> tuple[BenchmarkMetrics, list[TaskState]]:
        case_list = list(cases)
        states: list[TaskState] = []
        matches = 0
        for case in case_list:
            adapter = ReplayAdapter(case.outputs)
            state = self.engine_factory(adapter).run(case.task_id, case.request)
            states.append(state)
            if case.expected_contains is None:
                matches += int(state.status == "completed")
            else:
                matches += int(
                    state.status == "completed"
                    and state.answer is not None
                    and case.expected_contains in state.answer
                )

        count = len(states)
        completed = sum(state.status == "completed" for state in states)
        tool_steps = sum(len(state.history) for state in states)
        duplicates = sum(state.duplicate_actions for state in states)
        tool_errors = sum(
            record.observation.error is not None
            for state in states
            for record in state.history
        )
        metrics = BenchmarkMetrics(
            tasks=count,
            completed=completed,
            completion_rate=completed / count if count else 0.0,
            expected_matches=matches,
            expected_match_rate=matches / count if count else 0.0,
            terminated=count - completed,
            average_steps=fmean(state.step_count for state in states) if states else 0.0,
            average_latency_ms=fmean(state.total_latency_ms for state in states) if states else 0.0,
            repair_task_rate=(
                sum(state.repair_attempts > 0 for state in states) / count if count else 0.0
            ),
            duplicate_action_rate=duplicates / tool_steps if tool_steps else 0.0,
            tool_error_rate=tool_errors / tool_steps if tool_steps else 0.0,
        )
        return metrics, states


@dataclass(frozen=True)
class LiveBenchmarkCase:
    """One task for a real backend shared across the complete manifest.

    ``required_tools`` makes observation use measurable without claiming to
    inspect a model's hidden reasoning. A case counts as observation-using only
    when every named tool produced an error-free observation, a later model
    call received observation-bearing state, and the case met its external
    success check.
    """

    task_id: str
    request: str
    expected_contains: str | None = None
    required_tools: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LiveBenchmarkCase":
        required = {"task_id", "request"}
        optional = {"expected_contains", "required_tools"}
        missing = required - set(value)
        extra = set(value) - required - optional
        if missing or extra:
            raise ValueError(
                f"live benchmark case keys: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        task_id = value["task_id"]
        request = value["request"]
        expected = value.get("expected_contains")
        tools = value.get("required_tools", [])
        if not isinstance(task_id, str) or not task_id.strip():
            raise TypeError("live benchmark task_id must be a non-empty string")
        if not isinstance(request, str) or not request.strip():
            raise TypeError("live benchmark request must be a non-empty string")
        if expected is not None and (not isinstance(expected, str) or not expected):
            raise TypeError("expected_contains must be a non-empty string or null")
        if (
            not isinstance(tools, list)
            or not all(isinstance(item, str) and item for item in tools)
            or len(set(tools)) != len(tools)
        ):
            raise TypeError("required_tools must be a list of unique non-empty strings")
        return cls(task_id, request, expected, tuple(tools))

    def answer_matched(self, state: TaskState) -> bool:
        if state.status != "completed":
            return False
        if self.expected_contains is None:
            return True
        return (
            state.answer is not None
            and self.expected_contains.casefold() in state.answer.casefold()
        )

    def _required_observation_flow(self, state: TaskState) -> bool:
        if not self.required_tools:
            return False
        observed = {
            str(record.action.get("tool"))
            for record in state.history
            if record.observation.error is None
        }
        return (
            set(self.required_tools).issubset(observed)
            and state.observation_prompt_calls > 0
        )

    def succeeded(self, state: TaskState) -> bool:
        return self.answer_matched(state) and (
            not self.required_tools or self._required_observation_flow(state)
        )

    def used_required_observations(self, state: TaskState) -> bool:
        return (
            bool(self.required_tools)
            and self.answer_matched(state)
            and self._required_observation_flow(state)
        )


@dataclass(frozen=True)
class LiveBenchmarkMetrics:
    tasks: int
    task_successes: int
    task_success_rate: float
    completed: int
    terminated: int
    total_model_calls: int
    average_model_calls: float
    strict_valid_actions: int
    strict_action_validity_rate: float
    repair_attempts: int
    repair_recoveries: int
    repair_recovery_rate: float
    observation_required_tasks: int
    observation_used_tasks: int
    observation_use_rate: float
    average_steps: float
    average_tool_calls: float
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    duplicate_action_rate: float
    tool_error_rate: float
    termination_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in data.items()
        }


class LiveBenchmarkRunner:
    """Run a manifest through one shared, lazily loaded model adapter."""

    def __init__(
        self,
        adapter: ModelAdapter,
        engine_factory: Callable[[ModelAdapter], AgentEngine],
    ) -> None:
        self.adapter = adapter
        self.engine_factory = engine_factory

    def run(
        self, cases: Iterable[LiveBenchmarkCase]
    ) -> tuple[LiveBenchmarkMetrics, list[TaskState]]:
        case_list = list(cases)
        if len({case.task_id for case in case_list}) != len(case_list):
            raise ValueError("live benchmark task_id values must be unique")
        states = [
            self.engine_factory(self.adapter).run(case.task_id, case.request)
            for case in case_list
        ]
        count = len(states)
        successes = sum(case.succeeded(state) for case, state in zip(case_list, states))
        completed = sum(state.status == "completed" for state in states)
        model_calls = sum(state.model_calls for state in states)
        valid_actions = sum(state.valid_actions for state in states)
        repair_attempts = sum(state.repair_attempts for state in states)
        repair_recoveries = sum(state.repair_successes for state in states)
        required_cases = sum(bool(case.required_tools) for case in case_list)
        used_observations = sum(
            case.used_required_observations(state)
            for case, state in zip(case_list, states)
        )
        tool_calls = sum(len(state.history) for state in states)
        duplicate_actions = sum(state.duplicate_actions for state in states)
        tool_errors = sum(
            record.observation.error is not None
            for state in states
            for record in state.history
        )
        latencies = [state.total_latency_ms for state in states]
        ordered_latencies = sorted(latencies)
        p95_latency = (
            ordered_latencies[max(0, ceil(0.95 * count) - 1)] if count else 0.0
        )
        terminations: dict[str, int] = {}
        for state in states:
            reason = state.termination_reason or state.status
            terminations[reason] = terminations.get(reason, 0) + 1

        metrics = LiveBenchmarkMetrics(
            tasks=count,
            task_successes=successes,
            task_success_rate=successes / count if count else 0.0,
            completed=completed,
            terminated=count - completed,
            total_model_calls=model_calls,
            average_model_calls=model_calls / count if count else 0.0,
            strict_valid_actions=valid_actions,
            strict_action_validity_rate=valid_actions / model_calls if model_calls else 0.0,
            repair_attempts=repair_attempts,
            repair_recoveries=repair_recoveries,
            repair_recovery_rate=(
                repair_recoveries / repair_attempts if repair_attempts else 0.0
            ),
            observation_required_tasks=required_cases,
            observation_used_tasks=used_observations,
            observation_use_rate=(
                used_observations / required_cases if required_cases else 0.0
            ),
            average_steps=fmean(state.step_count for state in states) if states else 0.0,
            average_tool_calls=tool_calls / count if count else 0.0,
            average_latency_ms=fmean(latencies) if latencies else 0.0,
            median_latency_ms=median(latencies) if latencies else 0.0,
            p95_latency_ms=p95_latency,
            duplicate_action_rate=(
                duplicate_actions / tool_calls if tool_calls else 0.0
            ),
            tool_error_rate=tool_errors / tool_calls if tool_calls else 0.0,
            termination_counts=dict(sorted(terminations.items())),
        )
        return metrics, states
