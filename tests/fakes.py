from __future__ import annotations

from collections.abc import Callable

from dllm_final.types import TaskState


class FakeAdapter:
    def __init__(self, function: Callable[[str, TaskState, int], str]) -> None:
        self.function = function
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, *, prompt: str, state: TaskState) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.function(prompt, state, self.calls)
