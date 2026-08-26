"""Bounded observe-act-update loop with strict repair and termination rules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any

from .actions import ActionParseError, StrictActionParser
from .adapters import ModelAdapter
from .eventlog import TaskJSONLLogger
from .tools import ToolExecutor
from .types import Observation, TaskState


@dataclass(frozen=True)
class EngineConfig:
    max_steps: int = 8
    max_no_progress: int = 2
    prompt_observations: int = 6

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_no_progress < 1:
            raise ValueError("max_no_progress must be at least 1")
        if self.prompt_observations < 0:
            raise ValueError("prompt_observations cannot be negative")


class AgentEngine:
    def __init__(
        self,
        model: ModelAdapter,
        executor: ToolExecutor,
        *,
        parser: StrictActionParser | None = None,
        config: EngineConfig | None = None,
        logger: TaskJSONLLogger | None = None,
    ) -> None:
        self.model = model
        self.executor = executor
        self.parser = parser or StrictActionParser()
        self.config = config or EngineConfig()
        self.logger = logger

    def run(self, task_id: str, request: str) -> TaskState:
        state = TaskState(task_id=task_id, request=request)
        # De-duplication is scoped to this run. Reusing an engine for another
        # task must never turn its first observation into a cache hit.
        execution_cache: dict[str, Observation] = {}
        self._log(
            state,
            "task_started",
            payload={
                "max_steps": self.config.max_steps,
                "max_no_progress": self.config.max_no_progress,
                "tools": self.executor.registry.names(),
            },
        )

        while state.status == "running" and state.step_count < self.config.max_steps:
            prompt = self._build_prompt(state)
            try:
                raw = self._call_model(state, prompt, phase="action")
            except Exception as exc:
                self._terminate(state, "model_error", error=exc)
                break

            def repair(repair_prompt: str) -> str:
                state.record_repair()
                return self._call_model(state, repair_prompt, phase="repair")

            try:
                outcome = self.parser.parse_with_one_repair(raw, repair)
            except ActionParseError as exc:
                self._terminate(state, "invalid_action", error=exc)
                break
            except Exception as exc:
                # The parser normalizes validation failures to ActionParseError;
                # another exception here came from the repair backend call.
                self._terminate(state, "model_error", error=exc)
                break

            action = outcome.action
            state.record_valid_action(repaired=outcome.repaired)
            self._log(
                state,
                "action_parsed",
                payload={
                    "action": action.to_dict(),
                    "action_hash": action.canonical_hash(),
                    "repaired": outcome.repaired,
                },
            )

            if action.kind == "final":
                state.apply_final(action)
                self._log(state, "task_completed", payload={"answer": state.answer})
                break

            result = self.executor.execute(action, cache=execution_cache)
            progress = state.apply_tool_result(
                action,
                result.observation,
                action_hash=result.action_hash,
                cached=result.cached,
                latency_ms=result.latency_ms,
            )
            self._log(
                state,
                "tool_observation",
                latency_ms=result.latency_ms,
                payload={
                    "action_hash": result.action_hash,
                    "cached": result.cached,
                    "progress": progress,
                    "summary": result.observation.summary,
                    "raw_reference": result.observation.raw_reference,
                    "error": result.observation.error,
                    "reliability": result.observation.reliability,
                },
            )

            if state.no_progress_count >= self.config.max_no_progress:
                self._terminate(state, "no_progress")
                break

        if state.status == "running":
            self._terminate(state, "max_steps")
        return state

    def _build_prompt(self, state: TaskState) -> str:
        catalog = json.dumps(
            self.executor.registry.prompt_catalog(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        snapshot = state.to_prompt_snapshot(max_observations=self.config.prompt_observations)
        observation_policy = ""
        if state.history:
            observation_policy = (
                "A tool observation is available. If it answers the request, return a final action "
                "now using its exact result. Do not repeat an identical tool call. If it does not "
                "answer, choose a different action.\n"
            )
        return (
            "You are the decision component of a bounded tool agent. Return exactly one JSON object. "
            "Do not use Markdown or add commentary.\n"
            f"{observation_policy}"
            f"Action schema: {self.parser.schema}\n"
            f"Available tools: {catalog}\n"
            f"Current task state: {snapshot}"
        )

    def _call_model(self, state: TaskState, prompt: str, *, phase: str) -> str:
        state.record_model_call(
            includes_observation=phase == "action" and bool(state.history)
        )
        started = perf_counter()
        try:
            raw = self.model.generate(prompt=prompt, state=state)
        except Exception as exc:
            latency_ms = (perf_counter() - started) * 1_000
            state.record_model_latency(latency_ms)
            self._log(
                state,
                "model_call_failed",
                latency_ms=latency_ms,
                payload={"phase": phase, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        latency_ms = (perf_counter() - started) * 1_000
        state.record_model_latency(latency_ms)
        self._log(
            state,
            "model_output",
            latency_ms=latency_ms,
            payload={"phase": phase, "characters": len(raw) if isinstance(raw, str) else None},
        )
        return raw

    def _terminate(self, state: TaskState, reason: str, *, error: Exception | None = None) -> None:
        state.terminate(reason)
        payload: dict[str, Any] = {"reason": reason}
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
        self._log(state, "task_terminated", payload=payload)

    def _log(
        self,
        state: TaskState,
        event: str,
        *,
        latency_ms: float = 0.0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.logger is not None:
            self.logger.log(
                state,
                event,
                latency_ms=latency_ms,
                payload=payload,
            )
