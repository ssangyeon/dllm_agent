from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from dllm_final.adapters import ReplayAdapter
from dllm_final.benchmark import BenchmarkCase, BenchmarkRunner
from dllm_final.cli import main
from dllm_final.engine import AgentEngine
from dllm_final.tools import ToolExecutor, default_registry


class BenchmarkTests(unittest.TestCase):
    def test_metrics_cover_success_repair_dedup_errors_and_latency(self) -> None:
        cases = [
            BenchmarkCase(
                "calc",
                "2+2",
                (
                    '{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}',
                    '{"type":"final","answer":"4"}',
                ),
                "4",
            ),
            BenchmarkCase(
                "repair",
                "answer",
                ("invalid", '{"type":"final","answer":"fixed"}'),
                "fixed",
            ),
        ]

        def factory(adapter: ReplayAdapter) -> AgentEngine:
            return AgentEngine(adapter, ToolExecutor(default_registry()))

        metrics, states = BenchmarkRunner(factory).run(cases)
        self.assertEqual(metrics.tasks, 2)
        self.assertEqual(metrics.completed, 2)
        self.assertEqual(metrics.completion_rate, 1.0)
        self.assertEqual(metrics.expected_match_rate, 1.0)
        self.assertEqual(metrics.repair_task_rate, 0.5)
        self.assertEqual(metrics.duplicate_action_rate, 0.0)
        self.assertEqual(metrics.tool_error_rate, 0.0)
        self.assertGreaterEqual(metrics.average_latency_ms, 0.0)
        self.assertEqual([state.status for state in states], ["completed", "completed"])

    def test_empty_benchmark_is_well_defined(self) -> None:
        metrics, states = BenchmarkRunner(
            lambda adapter: AgentEngine(adapter, ToolExecutor(default_registry()))
        ).run([])
        self.assertEqual(states, [])
        self.assertEqual(metrics.tasks, 0)
        self.assertEqual(metrics.completion_rate, 0.0)


class CliIntegrationTests(unittest.TestCase):
    def test_demo_command_runs_complete_loop(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = main(["demo", "--query", "7 * 8", "--task-id", "cli-test"])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["answer"], "Calculation result: 56")

    def test_demo_retrieves_from_configured_local_docs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text(
                "Explicit state update prevents hidden transitions.", encoding="utf-8"
            )
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "demo",
                        "--query",
                        "state update",
                        "--docs",
                        str(root),
                        "--task-id",
                        "retrieval-test",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertIn("guide.md:1", payload["answer"])

    def test_benchmark_command_reads_replay_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "one",
                            "request": "done",
                            "outputs": ['{"type":"final","answer":"done"}'],
                            "expected_contains": "done",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                code = main(["benchmark", str(path)])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["metrics"]["completion_rate"], 1.0)

    def test_cli_reports_bad_adapter_config_without_traceback(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "demo",
                    "--adapter",
                    "x:y",
                    "--adapter-config",
                    "[]",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("JSON object", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
