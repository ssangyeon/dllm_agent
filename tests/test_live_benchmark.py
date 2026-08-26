from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from dllm_final.benchmark import LiveBenchmarkCase, LiveBenchmarkRunner
from dllm_final.cli import main
from dllm_final.engine import AgentEngine
from dllm_final.tools import ToolExecutor, default_registry
from dllm_final.types import TaskState


class PerTaskAdapter:
    def __init__(self, outputs: dict[str, list[str]]) -> None:
        self.outputs = {key: iter(value) for key, value in outputs.items()}
        self.calls: list[str] = []

    def generate(self, *, prompt: str, state: TaskState) -> str:
        del prompt
        self.calls.append(state.task_id)
        return next(self.outputs[state.task_id])


class LiveBenchmarkTests(unittest.TestCase):
    def test_shared_adapter_metrics_are_call_grounded(self) -> None:
        cases = [
            LiveBenchmarkCase("calc", "calculate", "42", ("calculator",)),
            LiveBenchmarkCase("repair", "repair", "fixed"),
            LiveBenchmarkCase("invalid", "fail", "never"),
        ]
        adapter = PerTaskAdapter(
            {
                "calc": [
                    '{"type":"tool","tool":"calculator","arguments":{"expression":"6*7"}}',
                    '{"type":"final","answer":"42"}',
                ],
                "repair": ["invalid prose", '{"type":"final","answer":"fixed"}'],
                "invalid": ["bad", "still bad"],
            }
        )

        def factory(model: PerTaskAdapter) -> AgentEngine:
            return AgentEngine(model, ToolExecutor(default_registry()))

        metrics, states = LiveBenchmarkRunner(adapter, factory).run(cases)
        self.assertEqual(len(states), 3)
        self.assertEqual(metrics.task_successes, 2)
        self.assertAlmostEqual(metrics.task_success_rate, 2 / 3)
        self.assertEqual(metrics.total_model_calls, 6)
        self.assertEqual(metrics.strict_valid_actions, 3)
        self.assertEqual(metrics.strict_action_validity_rate, 0.5)
        self.assertEqual(metrics.repair_attempts, 2)
        self.assertEqual(metrics.repair_recoveries, 1)
        self.assertEqual(metrics.repair_recovery_rate, 0.5)
        self.assertEqual(metrics.observation_required_tasks, 1)
        self.assertEqual(metrics.observation_used_tasks, 1)
        self.assertEqual(metrics.observation_use_rate, 1.0)
        self.assertEqual(metrics.termination_counts, {"invalid_action": 1, "model_final": 2})
        self.assertEqual(adapter.calls, ["calc", "calc", "repair", "repair", "invalid", "invalid"])

    def test_manifest_validation_is_strict(self) -> None:
        with self.assertRaises(ValueError):
            LiveBenchmarkCase.from_dict({"task_id": "x", "request": "y", "extra": 1})
        with self.assertRaises(TypeError):
            LiveBenchmarkCase.from_dict(
                {"task_id": "x", "request": "y", "required_tools": ["retrieve", "retrieve"]}
            )

    def test_required_tool_cannot_be_bypassed_by_matching_answer(self) -> None:
        case = LiveBenchmarkCase("direct", "calculate", "42", ("calculator",))
        adapter = PerTaskAdapter(
            {"direct": ['{"type":"final","answer":"42"}']}
        )

        def factory(model: PerTaskAdapter) -> AgentEngine:
            return AgentEngine(model, ToolExecutor(default_registry()))

        metrics, states = LiveBenchmarkRunner(adapter, factory).run([case])
        self.assertFalse(case.succeeded(states[0]))
        self.assertEqual(metrics.task_success_rate, 0.0)
        self.assertEqual(metrics.observation_use_rate, 0.0)

    def test_runner_rejects_duplicate_task_ids_before_model_calls(self) -> None:
        adapter = PerTaskAdapter({"same": ['{"type":"final","answer":"ok"}']})

        def factory(model: PerTaskAdapter) -> AgentEngine:
            return AgentEngine(model, ToolExecutor(default_registry()))

        with self.assertRaisesRegex(ValueError, "unique"):
            LiveBenchmarkRunner(adapter, factory).run(
                [
                    LiveBenchmarkCase("same", "one"),
                    LiveBenchmarkCase("same", "two"),
                ]
            )
        self.assertEqual(adapter.calls, [])

    def test_live_cli_loads_one_lazy_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "live.json"
            manifest.write_text(
                json.dumps(
                    [{"task_id": "one", "request": "finish", "expected_contains": "DONE"}]
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "live-benchmark",
                        str(manifest),
                        "--adapter",
                        "tests.live_backend_plugin:create",
                        "--backend-kind",
                        "custom",
                        "--adapter-config",
                        '{"answer":"done"}',
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["backend"]["loaded"])
            self.assertEqual(payload["metrics"]["task_success_rate"], 1.0)
            self.assertEqual(payload["metrics"]["strict_action_validity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
