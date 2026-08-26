from __future__ import annotations

import unittest

from dllm_final.actions import ActionParseError, StrictActionParser


class StrictActionParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = StrictActionParser()

    def test_parses_exact_tool_and_final_schemas(self) -> None:
        tool = self.parser.parse('{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}')
        final = self.parser.parse(' {"type":"final","answer":"  done  "} ')
        self.assertEqual(tool.tool, "calculator")
        self.assertEqual(tool.arguments, {"expression": "2+2"})
        self.assertEqual(final.answer, "done")

    def test_canonical_hash_ignores_object_key_order(self) -> None:
        first = self.parser.parse(
            '{"type":"tool","tool":"x","arguments":{"b":2,"a":1}}'
        )
        second = self.parser.parse(
            '{"arguments":{"a":1,"b":2},"tool":"x","type":"tool"}'
        )
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.canonical_hash(), second.canonical_hash())

    def test_rejects_markdown_unknown_keys_and_bad_tool_name(self) -> None:
        invalid = [
            '```json\n{"type":"final","answer":"x"}\n```',
            '{"type":"final","answer":"x","thought":"hidden"}',
            '{"type":"tool","tool":"bad name","arguments":{}}',
            '{"type":"tool","tool":"x"}',
            '[]',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ActionParseError):
                self.parser.parse(raw)

    def test_rejects_duplicate_keys_at_any_depth(self) -> None:
        with self.assertRaisesRegex(ActionParseError, "duplicate"):
            self.parser.parse(
                '{"type":"tool","tool":"x","arguments":{"a":1,"a":2}}'
            )

    def test_rejects_non_finite_and_excessive_nesting(self) -> None:
        for number in ("NaN", "Infinity", "1e9999"):
            with self.subTest(number=number), self.assertRaises(ActionParseError):
                self.parser.parse(
                    '{"type":"tool","tool":"x","arguments":{"n":' + number + "}}"
                )
        nested = "[]"
        for _ in range(14):
            nested = "[" + nested + "]"
        with self.assertRaisesRegex(ActionParseError, "nesting"):
            self.parser.parse(
                '{"type":"tool","tool":"x","arguments":{"n":' + nested + "}}"
            )

    def test_valid_output_does_not_call_repair(self) -> None:
        calls = 0

        def repair(_: str) -> str:
            nonlocal calls
            calls += 1
            return "{}"

        outcome = self.parser.parse_with_one_repair(
            '{"type":"final","answer":"ok"}', repair
        )
        self.assertFalse(outcome.repaired)
        self.assertEqual(calls, 0)

    def test_invalid_output_gets_exactly_one_targeted_repair(self) -> None:
        prompts: list[str] = []

        def repair(prompt: str) -> str:
            prompts.append(prompt)
            return '{"type":"final","answer":"fixed"}'

        outcome = self.parser.parse_with_one_repair("not-json", repair)
        self.assertTrue(outcome.repaired)
        self.assertEqual(outcome.action.answer, "fixed")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Validation error", prompts[0])
        self.assertIn("Invalid output", prompts[0])

    def test_failed_repair_is_not_retried(self) -> None:
        calls = 0

        def repair(_: str) -> str:
            nonlocal calls
            calls += 1
            return "still invalid"

        with self.assertRaisesRegex(ActionParseError, "one attempt"):
            self.parser.parse_with_one_repair("invalid", repair)
        self.assertEqual(calls, 1)

    def test_parser_recursion_limit_still_uses_one_repair(self) -> None:
        nested = "[" * 1_100 + "0" + "]" * 1_100
        raw = '{"type":"tool","tool":"x","arguments":{"n":' + nested + "}}"
        calls = 0

        def repair(_: str) -> str:
            nonlocal calls
            calls += 1
            return '{"type":"final","answer":"recovered"}'

        outcome = self.parser.parse_with_one_repair(raw, repair)
        self.assertEqual(outcome.action.answer, "recovered")
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
