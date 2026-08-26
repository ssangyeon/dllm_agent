from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from dllm_final.actions import StrictActionParser
from dllm_final.tools import (
    KeywordRetriever,
    SafeCalculator,
    ToolExecutor,
    ToolRegistry,
    calculator_tool,
    retrieval_tool,
    ToolSpec,
)
from dllm_final.types import Observation


class CalculatorTests(unittest.TestCase):
    def test_operator_precedence_and_numeric_types(self) -> None:
        calculator = SafeCalculator()
        self.assertEqual(calculator.evaluate("2 + 3 * 4"), 14)
        self.assertEqual(calculator.evaluate("-(9 // 2) + 10 % 4"), -2)
        self.assertAlmostEqual(calculator.evaluate("5 / 2"), 2.5)

    def test_forbids_code_execution_syntax(self) -> None:
        calculator = SafeCalculator()
        attacks = [
            "__import__('os').system('id')",
            "open('/etc/passwd')",
            "(1).__class__",
            "[1, 2][0]",
            "True + 1",
            "2 ^ 3",
        ]
        for expression in attacks:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator.evaluate(expression)

    def test_bounds_exponents_size_nodes_and_arithmetic_errors(self) -> None:
        calculator = SafeCalculator(max_chars=40, max_nodes=12, max_exponent=5)
        for expression in ("2**6", "1/0", "1" * 41, "+".join("1" for _ in range(10))):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator.evaluate(expression)

    def test_executor_normalizes_errors_and_caches_canonical_action(self) -> None:
        parser = StrictActionParser()
        executor = ToolExecutor(self._registry())
        first = parser.parse(
            '{"type":"tool","tool":"calculator","arguments":{"expression":"1/0"}}'
        )
        second = parser.parse(
            '{"arguments":{"expression":"1/0"},"tool":"calculator","type":"tool"}'
        )
        result1 = executor.execute(first)
        result2 = executor.execute(second)
        self.assertIsNotNone(result1.observation.error)
        self.assertFalse(result1.cached)
        self.assertTrue(result2.cached)
        self.assertEqual(result1.action_hash, result2.action_hash)
        self.assertEqual(executor.cache_size, 1)

    def test_tool_cannot_mutate_hashed_action_arguments(self) -> None:
        registry = ToolRegistry()

        def validate(arguments: object) -> None:
            return None

        def mutate(arguments: object) -> Observation:
            assert isinstance(arguments, dict)
            arguments["nested"]["value"] = 99
            return Observation("ok", "mutation-test", None, 1.0)

        registry.register(ToolSpec("mutate", "test", mutate, validate))  # type: ignore[arg-type]
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"mutate","arguments":{"nested":{"value":1}}}'
        )
        before = action.canonical_hash()
        ToolExecutor(registry).execute(action)
        self.assertEqual(action.arguments, {"nested": {"value": 1}})
        self.assertEqual(action.canonical_hash(), before)

    def test_non_cacheable_transient_failure_is_retried(self) -> None:
        registry = ToolRegistry()
        calls = 0

        def validate(arguments: object) -> None:
            return None

        def transient(arguments: object) -> Observation:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary")
            return Observation("recovered", "transient", None, 1.0)

        registry.register(
            ToolSpec("transient", "test", transient, validate, cacheable=False)  # type: ignore[arg-type]
        )
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"transient","arguments":{}}'
        )
        executor = ToolExecutor(registry)
        first = executor.execute(action)
        second = executor.execute(action)
        self.assertIsNotNone(first.observation.error)
        self.assertFalse(first.cached)
        self.assertEqual(second.observation.summary, "recovered")
        self.assertFalse(second.cached)
        self.assertEqual(calls, 2)
        self.assertEqual(executor.cache_size, 0)

    def test_tool_metadata_is_bounded_for_prompt_safety(self) -> None:
        with self.assertRaisesRegex(ValueError, "description exceeds"):
            ToolSpec("tool", "x" * 513, lambda args: Observation("x"), lambda args: None)
        registry = ToolRegistry()
        for index in range(64):
            registry.register(
                ToolSpec(
                    f"tool{index}",
                    "test",
                    lambda args: Observation("x"),
                    lambda args: None,
                )
            )
        with self.assertRaisesRegex(ValueError, "64 tools"):
            registry.register(
                ToolSpec("overflow", "test", lambda args: Observation("x"), lambda args: None)
            )

    @staticmethod
    def _registry() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(calculator_tool())
        return registry


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "b.txt").write_text("agent loop alpha\nother", encoding="utf-8")
        (self.root / "a.md").write_text("agent loop beta\n에이전트 상태 갱신", encoding="utf-8")
        (self.root / "ignored.py").write_text("agent loop should not appear", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_search_is_deterministic_with_path_line_tiebreak(self) -> None:
        retriever = KeywordRetriever(self.root)
        first = retriever.search("agent loop", top_k=10)
        second = retriever.search("agent loop", top_k=10)
        self.assertEqual(first, second)
        self.assertEqual([hit.relative_path for hit in first], ["a.md", "b.txt"])
        self.assertNotIn("ignored.py", [hit.relative_path for hit in first])

    def test_unicode_tokens_and_structured_references(self) -> None:
        registry = ToolRegistry()
        registry.register(retrieval_tool(KeywordRetriever(self.root)))
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"retrieve","arguments":{"query":"상태 갱신","top_k":2}}'
        )
        observation = ToolExecutor(registry).execute(action).observation
        references = json.loads(observation.raw_reference or "[]")
        self.assertIn("에이전트 상태 갱신", observation.summary)
        self.assertEqual(references[0]["path"], "a.md")
        self.assertGreater(observation.reliability, 0.5)

    def test_no_match_is_not_an_execution_error(self) -> None:
        registry = ToolRegistry()
        registry.register(retrieval_tool(KeywordRetriever(self.root)))
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"retrieve","arguments":{"query":"absent"}}'
        )
        observation = ToolExecutor(registry).execute(action).observation
        self.assertIsNone(observation.error)
        self.assertIn("No keyword matches", observation.summary)
        self.assertEqual(observation.reliability, 0.2)

    def test_retriever_does_not_follow_file_symlink_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "secret.txt"
            outside.write_text("secret escape token", encoding="utf-8")
            try:
                (self.root / "linked.txt").symlink_to(outside)
            except OSError:
                self.skipTest("symlinks unavailable")
            self.assertEqual(KeywordRetriever(self.root).search("secret escape"), [])

    def test_retrieval_argument_validation_is_strict(self) -> None:
        registry = ToolRegistry()
        registry.register(retrieval_tool(KeywordRetriever(self.root)))
        executor = ToolExecutor(registry)
        action = StrictActionParser().parse(
            '{"type":"tool","tool":"retrieve","arguments":{"query":"agent","path":"/tmp"}}'
        )
        observation = executor.execute(action).observation
        self.assertIn("wrong arguments", observation.error or "")

    def test_query_and_corpus_resource_limits_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integers"):
            KeywordRetriever(self.root, max_files=0)
        retriever = KeywordRetriever(self.root, max_query_chars=5)
        with self.assertRaisesRegex(ValueError, "query exceeds"):
            retriever.search("agent loop")

        (self.root / "lines.txt").write_text("agent\nagent\nagent\n", encoding="utf-8")
        line_limited = KeywordRetriever(self.root, max_lines=2)
        with self.assertRaisesRegex(ValueError, "lines exceed"):
            line_limited.search("agent")

    def test_streaming_top_k_does_not_accumulate_all_matches(self) -> None:
        (self.root / "many.txt").write_text(
            "\n".join(f"agent line {number}" for number in range(100)),
            encoding="utf-8",
        )
        hits = KeywordRetriever(self.root).search("agent", top_k=3)
        self.assertEqual(len(hits), 3)
        self.assertEqual(
            [(hit.relative_path, hit.line_number) for hit in hits],
            [("a.md", 1), ("b.txt", 1), ("many.txt", 1)],
        )


if __name__ == "__main__":
    unittest.main()
