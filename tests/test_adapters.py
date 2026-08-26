from __future__ import annotations

import sys
import types
import unittest

from dllm_final.adapters import LazyBackendAdapter, ReplayAdapter
from dllm_final.types import TaskState


class AdapterTests(unittest.TestCase):
    def test_replay_adapter_is_deterministic_and_bounded(self) -> None:
        adapter = ReplayAdapter(["one", "two"])
        state = TaskState("id", "request")
        self.assertEqual(adapter.generate(prompt="a", state=state), "one")
        self.assertEqual(adapter.generate(prompt="b", state=state), "two")
        with self.assertRaisesRegex(RuntimeError, "no output left"):
            adapter.generate(prompt="c", state=state)
        self.assertEqual(adapter.calls, ["a", "b", "c"])

    def test_optional_backend_is_imported_lazily_once(self) -> None:
        module_name = "_dllm_final_test_plugin"
        module = types.ModuleType(module_name)
        factory_calls: list[str] = []

        class Plugin:
            def __init__(self, prefix: str) -> None:
                self.prefix = prefix

            def generate(self, *, prompt: str, state: TaskState) -> str:
                return self.prefix + prompt

        def factory(prefix: str) -> Plugin:
            factory_calls.append(prefix)
            return Plugin(prefix)

        module.factory = factory  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        try:
            adapter = LazyBackendAdapter(
                f"{module_name}:factory", backend_kind="dllm", config={"prefix": "!"}
            )
            self.assertFalse(adapter.loaded)
            state = TaskState("id", "request")
            self.assertEqual(adapter.generate(prompt="x", state=state), "!x")
            self.assertEqual(adapter.generate(prompt="y", state=state), "!y")
            self.assertTrue(adapter.loaded)
            self.assertEqual(factory_calls, ["!"])
        finally:
            sys.modules.pop(module_name, None)

    def test_backend_kind_and_factory_path_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            LazyBackendAdapter("missing_separator")
        with self.assertRaises(ValueError):
            LazyBackendAdapter("a:b", backend_kind="unknown")


if __name__ == "__main__":
    unittest.main()
