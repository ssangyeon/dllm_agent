from __future__ import annotations

import json


class LivePlugin:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, *, prompt: str, state: object) -> str:
        del prompt, state
        return json.dumps({"type": "final", "answer": self.answer})


def create(*, answer: str = "done") -> LivePlugin:
    return LivePlugin(answer)
