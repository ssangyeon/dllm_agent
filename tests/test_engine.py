from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dllm_final.actions import StrictActionParser
from dllm_final.adapters import ReplayAdapter
from dllm_final.engine import AgentEngine, EngineConfig
from dllm_final.eventlog import TaskJSONLLogger
from dllm_final.tools import (
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    calculator_tool,
)
from dllm_final.types import Observation
from tests.fakes import FakeAdapter


def calculator_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(calculator_tool())
    return ToolExecutor(registry)


class EngineTests(unittest.TestCase):
    def test_prompt_tells_model_to_finalize_from_observation_and_not_repeat(self) -> None:
        adapter = ReplayAdapter([
            '{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}',
            '{"type":"final","answer":"4"}',
        ])
        state = AgentEngine(adapter, calculator_executor()).run("prompt-policy", "2+2")
        self.assertEqual(state.status, "completed")
        self.assertNotIn("A tool observation is available", adapter.calls[0])
        self.assertIn("return a final action now using its exact result", adapter.calls[1])
        self.assertIn("Do not repeat an identical tool call", adapter.calls[1])

    def test_end_to_end_tool_observation_and_final(self) -> None:
        adapter = ReplayAdapter(
            [
                '{"type":"tool","tool":"calculator","arguments":{"expression":"6*7"}}',
                '{"type":"final","answer":"The answer is 42."}',
            ]
        )
        state = AgentEngine(adapter, calculator_executor()).run("calc", "what is 6*7?")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.answer, "The answer is 42.")
        self.assertEqual(state.step_count, 2)
        self.assertEqual(state.history[0].observation.summary, "Calculation result: 42")
        self.assertTrue(state.history[0].progress)
        self.assertIn("Calculation result: 42", adapter.calls[1])

    def test_engine_makes_exactly_one_repair_call(self) -> None:
        adapter = ReplayAdapter(
            ["answer in prose", '{"type":"final","answer":"repaired"}']
        )
        state = AgentEngine(adapter, calculator_executor()).run("repair", "request")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.repair_attempts, 1)
        self.assertEqual(len(adapter.calls), 2)
        self.assertIn("Repair only", adapter.calls[1])

    def test_failed_repair_terminates_without_a_third_call(self) -> None:
        adapter = ReplayAdapter(["invalid", "also invalid", '{"type":"final","answer":"late"}'])
        state = AgentEngine(adapter, calculator_executor()).run("bad-repair", "request")
        self.assertEqual(state.status, "terminated")
        self.assertEqual(state.termination_reason, "invalid_action")
        self.assertEqual(state.repair_attempts, 1)
        self.assertEqual(len(adapter.calls), 2)

    def test_backend_failure_during_repair_is_a_model_error(self) -> None:
        def fail_repair(prompt: str, state: object, calls: int) -> str:
            if calls == 1:
                return "invalid"
            raise RuntimeError("repair backend offline")

        adapter = FakeAdapter(fail_repair)  # type: ignore[arg-type]
        state = AgentEngine(adapter, calculator_executor()).run("repair-offline", "request")
        self.assertEqual(state.status, "terminated")
        self.assertEqual(state.termination_reason, "model_error")
        self.assertEqual(state.repair_attempts, 1)
        self.assertEqual(adapter.calls, 2)

    def test_duplicate_canonical_action_uses_cache_and_stops_no_progress(self) -> None:
        first = '{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}'
        reordered = '{"arguments":{"expression":"2+2"},"tool":"calculator","type":"tool"}'
        adapter = ReplayAdapter([first, reordered])
        engine = AgentEngine(
            adapter,
            calculator_executor(),
            config=EngineConfig(max_steps=8, max_no_progress=1),
        )
        state = engine.run("dedup", "calculate")
        self.assertEqual(state.termination_reason, "no_progress")
        self.assertEqual(state.step_count, 2)
        self.assertEqual(state.duplicate_actions, 1)
        self.assertTrue(state.history[1].cached)
        self.assertFalse(state.history[1].progress)

    def test_reusing_engine_does_not_share_cache_between_runs(self) -> None:
        tool = '{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}'
        final = '{"type":"final","answer":"4"}'
        adapter = ReplayAdapter([tool, final, tool, final])
        engine = AgentEngine(
            adapter,
            calculator_executor(),
            config=EngineConfig(max_no_progress=1),
        )
        first = engine.run("first", "calculate")
        second = engine.run("second", "calculate")
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "completed")
        self.assertFalse(first.history[0].cached)
        self.assertFalse(second.history[0].cached)

    def test_same_observation_from_distinct_actions_is_no_progress(self) -> None:
        registry = ToolRegistry()

        def validate(arguments: object) -> None:
            return None

        def constant(arguments: object) -> Observation:
            return Observation("unchanged", "constant", None, 0.5)

        registry.register(ToolSpec("constant", "constant", constant, validate))  # type: ignore[arg-type]
        adapter = ReplayAdapter(
            [
                '{"type":"tool","tool":"constant","arguments":{"n":1}}',
                '{"type":"tool","tool":"constant","arguments":{"n":2}}',
            ]
        )
        state = AgentEngine(
            adapter,
            ToolExecutor(registry),
            config=EngineConfig(max_no_progress=1),
        ).run("same", "request")
        self.assertEqual(state.termination_reason, "no_progress")
        self.assertFalse(state.history[1].cached)
        self.assertFalse(state.history[1].progress)

    def test_non_cacheable_poll_can_progress_with_a_new_observation(self) -> None:
        registry = ToolRegistry()
        polls = 0

        def validate(arguments: object) -> None:
            return None

        def poll(arguments: object) -> Observation:
            nonlocal polls
            polls += 1
            status = "pending" if polls == 1 else "ready"
            return Observation(status, "job-1", None, 1.0)

        registry.register(
            ToolSpec("poll", "poll job", poll, validate, cacheable=False)  # type: ignore[arg-type]
        )
        action = '{"type":"tool","tool":"poll","arguments":{"job":"1"}}'
        adapter = ReplayAdapter([action, action, '{"type":"final","answer":"ready"}'])
        state = AgentEngine(
            adapter,
            ToolExecutor(registry),
            config=EngineConfig(max_no_progress=1),
        ).run("poll", "wait for job")
        self.assertEqual(state.status, "completed")
        self.assertEqual([record.progress for record in state.history], [True, True])
        self.assertEqual([record.observation.summary for record in state.history], ["pending", "ready"])
        self.assertEqual(state.duplicate_actions, 1)

    def test_max_step_termination_is_hard_bound(self) -> None:
        adapter = ReplayAdapter(
            [
                '{"type":"tool","tool":"calculator","arguments":{"expression":"1+1"}}',
                '{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}',
                '{"type":"final","answer":"must not run"}',
            ]
        )
        state = AgentEngine(
            adapter,
            calculator_executor(),
            config=EngineConfig(max_steps=2, max_no_progress=2),
        ).run("bounded", "request")
        self.assertEqual(state.termination_reason, "max_steps")
        self.assertEqual(state.step_count, 2)
        self.assertEqual(len(adapter.calls), 2)

    def test_model_exception_is_contained(self) -> None:
        def fail(prompt: str, state: object, calls: int) -> str:
            raise RuntimeError("backend offline")

        adapter = FakeAdapter(fail)  # type: ignore[arg-type]
        state = AgentEngine(adapter, calculator_executor()).run("offline", "request")
        self.assertEqual(state.status, "terminated")
        self.assertEqual(state.termination_reason, "model_error")
        self.assertEqual(adapter.calls, 1)

    def test_failed_model_call_latency_is_logged(self) -> None:
        def fail(prompt: str, state: object, calls: int) -> str:
            raise RuntimeError("backend offline")

        with tempfile.TemporaryDirectory() as directory:
            logger = TaskJSONLLogger(directory)
            state = AgentEngine(
                FakeAdapter(fail),  # type: ignore[arg-type]
                calculator_executor(),
                logger=logger,
            ).run("failed-log", "request")
            records = [
                json.loads(line)
                for line in logger.path_for(state).read_text(encoding="utf-8").splitlines()
            ]
            failed = next(record for record in records if record["event"] == "model_call_failed")
            self.assertGreaterEqual(failed["latency_ms"], 0)
            self.assertEqual(failed["payload"]["phase"], "action")

    def test_unknown_tool_becomes_structured_observation(self) -> None:
        adapter = ReplayAdapter(
            [
                '{"type":"tool","tool":"missing","arguments":{}}',
                '{"type":"final","answer":"handled"}',
            ]
        )
        state = AgentEngine(adapter, calculator_executor()).run("unknown", "request")
        self.assertEqual(state.status, "completed")
        self.assertIn("unknown tool", state.history[0].observation.error or "")
        self.assertEqual(state.history[0].observation.reliability, 0.0)

    def test_jsonl_log_has_task_id_event_step_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = TaskJSONLLogger(directory)
            state = AgentEngine(
                ReplayAdapter(['{"type":"final","answer":"ok"}']),
                calculator_executor(),
                logger=logger,
            ).run("../../ unsafe id", "request")
            path = logger.path_for(state)
            self.assertEqual(path.parent, Path(directory))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["event"] for record in records], [
                "task_started", "model_output", "action_parsed", "task_completed"
            ])
            for record in records:
                self.assertEqual(record["task_id"], "../../ unsafe id")
                self.assertIn("step", record)
                self.assertGreaterEqual(record["latency_ms"], 0)
                self.assertIn("timestamp", record)


class TaskStateTests(unittest.TestCase):
    def test_state_rejects_empty_identifiers_and_invalid_transition(self) -> None:
        from dllm_final.types import TaskState

        with self.assertRaises(ValueError):
            TaskState("", "request")
        state = TaskState("id", "request")
        state.terminate("test")
        action = StrictActionParser().parse('{"type":"final","answer":"x"}')
        with self.assertRaises(RuntimeError):
            state.apply_final(action)

    def test_zero_prompt_observations_really_omits_history(self) -> None:
        from dllm_final.types import TaskState

        state = TaskState("id", "request")
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"calculator","arguments":{"expression":"1+1"}}'
        )
        result = calculator_executor().execute(action)
        state.apply_tool_result(
            action,
            result.observation,
            action_hash=result.action_hash,
            cached=result.cached,
            latency_ms=result.latency_ms,
        )
        snapshot = json.loads(state.to_prompt_snapshot(max_observations=0))
        self.assertEqual(snapshot["observations"], [])

    def test_request_and_observation_fields_are_bounded(self) -> None:
        from dllm_final.types import Observation, TaskState

        observation = Observation("s" * 10_000, "r" * 20_000, "e" * 5_000, 1)
        self.assertEqual(len(observation.summary), Observation.MAX_SUMMARY_CHARS)
        self.assertEqual(len(observation.raw_reference or ""), Observation.MAX_REFERENCE_CHARS)
        self.assertEqual(len(observation.error or ""), Observation.MAX_ERROR_CHARS)
        self.assertTrue(observation.summary.endswith("[truncated]"))
        self.assertEqual(observation.reliability, 1.0)
        with self.assertRaisesRegex(ValueError, "request exceeds"):
            TaskState("id", "q" * (TaskState.MAX_REQUEST_CHARS + 1))


if __name__ == "__main__":
    unittest.main()
