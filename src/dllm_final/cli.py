"""Command-line demo and replay benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .adapters import DemoAdapter, LazyBackendAdapter
from .benchmark import (
    BenchmarkCase,
    BenchmarkRunner,
    LiveBenchmarkCase,
    LiveBenchmarkRunner,
)
from .engine import AgentEngine, EngineConfig
from .eventlog import TaskJSONLLogger
from .tools import ToolExecutor, default_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dllm-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run one task with demo or optional plugin adapter")
    demo.add_argument("--query", default="2 + 3 * 4")
    demo.add_argument("--task-id")
    demo.add_argument("--docs", type=Path)
    demo.add_argument("--logs", type=Path)
    demo.add_argument("--adapter", help="lazy factory path: package.module:factory")
    demo.add_argument("--backend-kind", choices=("dllm", "ar", "custom"), default="custom")
    demo.add_argument("--adapter-config", default="{}", help="JSON object passed to adapter factory")
    demo.add_argument("--max-steps", type=int, default=8)
    demo.add_argument("--max-no-progress", type=int, default=2)

    benchmark = subparsers.add_parser("benchmark", help="run replay JSON cases")
    benchmark.add_argument("cases", type=Path)
    benchmark.add_argument("--docs", type=Path)
    benchmark.add_argument("--logs", type=Path)
    benchmark.add_argument("--max-steps", type=int, default=8)
    benchmark.add_argument("--max-no-progress", type=int, default=2)

    live = subparsers.add_parser(
        "live-benchmark", help="run one real lazy backend across a shared task manifest"
    )
    live.add_argument("cases", type=Path)
    live.add_argument("--adapter", required=True, help="lazy factory path: package.module:factory")
    live.add_argument("--backend-kind", choices=("dllm", "ar", "custom"), required=True)
    live.add_argument("--adapter-config", default="{}", help="JSON object passed to adapter factory")
    live.add_argument("--docs", type=Path)
    live.add_argument("--logs", type=Path)
    live.add_argument("--max-steps", type=int, default=8)
    live.add_argument("--max-no-progress", type=int, default=2)
    return parser


def _config(arguments: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        max_steps=arguments.max_steps,
        max_no_progress=arguments.max_no_progress,
    )


def _logger(path: Path | None) -> TaskJSONLLogger | None:
    return TaskJSONLLogger(path) if path is not None else None


def _load_json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("adapter config must be a JSON object")
    return value


def _run_demo(arguments: argparse.Namespace) -> int:
    registry = default_registry(documents=arguments.docs)
    if arguments.adapter:
        model = LazyBackendAdapter(
            arguments.adapter,
            backend_kind=arguments.backend_kind,
            config=_load_json_object(arguments.adapter_config),
        )
    else:
        model = DemoAdapter(retrieval_available=arguments.docs is not None)
    engine = AgentEngine(
        model,
        ToolExecutor(registry),
        config=_config(arguments),
        logger=_logger(arguments.logs),
    )
    task_id = arguments.task_id or datetime.now(timezone.utc).strftime("demo-%Y%m%dT%H%M%S.%fZ")
    state = engine.run(task_id, arguments.query)
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if state.status == "completed" else 2


def _run_benchmark(arguments: argparse.Namespace) -> int:
    raw = json.loads(arguments.cases.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("benchmark file must contain a JSON list")
    cases = [BenchmarkCase.from_dict(item) for item in raw if isinstance(item, dict)]
    if len(cases) != len(raw):
        raise TypeError("every benchmark list item must be an object")

    def factory(adapter: Any) -> AgentEngine:
        return AgentEngine(
            adapter,
            ToolExecutor(default_registry(documents=arguments.docs)),
            config=_config(arguments),
            logger=_logger(arguments.logs),
        )

    metrics, states = BenchmarkRunner(factory).run(cases)
    output = {
        "metrics": metrics.to_dict(),
        "tasks": [
            {
                "task_id": state.task_id,
                "status": state.status,
                "answer": state.answer,
                "termination_reason": state.termination_reason,
            }
            for state in states
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if metrics.terminated == 0 else 2


def _run_live_benchmark(arguments: argparse.Namespace) -> int:
    raw = json.loads(arguments.cases.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("live benchmark file must contain a JSON list")
    cases = [LiveBenchmarkCase.from_dict(item) for item in raw if isinstance(item, dict)]
    if len(cases) != len(raw):
        raise TypeError("every live benchmark list item must be an object")
    if not cases:
        raise ValueError("live benchmark file must contain at least one case")

    adapter = LazyBackendAdapter(
        arguments.adapter,
        backend_kind=arguments.backend_kind,
        config=_load_json_object(arguments.adapter_config),
    )

    def factory(model: Any) -> AgentEngine:
        return AgentEngine(
            model,
            ToolExecutor(default_registry(documents=arguments.docs)),
            config=_config(arguments),
            logger=_logger(arguments.logs),
        )

    metrics, states = LiveBenchmarkRunner(adapter, factory).run(cases)
    output = {
        "backend": {
            "factory": arguments.adapter,
            "kind": arguments.backend_kind,
            "loaded": adapter.loaded,
        },
        "metrics": metrics.to_dict(),
        "tasks": [
            {
                "task_id": state.task_id,
                "success": case.succeeded(state),
                "used_required_observations": case.used_required_observations(state),
                "status": state.status,
                "answer": state.answer,
                "termination_reason": state.termination_reason,
                "model_calls": state.model_calls,
                "valid_actions": state.valid_actions,
                "repair_attempts": state.repair_attempts,
                "repair_successes": state.repair_successes,
                "steps": state.step_count,
                "tool_calls": len(state.history),
                "latency_ms": round(state.total_latency_ms, 3),
            }
            for case, state in zip(cases, states)
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if metrics.task_successes == metrics.tasks else 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "demo":
            return _run_demo(arguments)
        if arguments.command == "benchmark":
            return _run_benchmark(arguments)
        return _run_live_benchmark(arguments)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
