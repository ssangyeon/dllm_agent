"""Backend-neutral model protocol and dependency-free adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import importlib
import json
import re
from typing import Any, Protocol, runtime_checkable

from .types import TaskState


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal protocol implemented by autoregressive or diffusion LMs."""

    def generate(self, *, prompt: str, state: TaskState) -> str:
        ...


class ReplayAdapter:
    """Deterministic adapter for tests, incident replay, and benchmarks."""

    def __init__(self, outputs: Iterable[str]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[str] = []

    def generate(self, *, prompt: str, state: TaskState) -> str:
        self.calls.append(prompt)
        try:
            return next(self._outputs)
        except StopIteration as exc:
            raise RuntimeError("replay adapter has no output left") from exc


class CallableAdapter:
    def __init__(self, function: Callable[[str, TaskState], str]) -> None:
        self.function = function

    def generate(self, *, prompt: str, state: TaskState) -> str:
        result = self.function(prompt, state)
        if not isinstance(result, str):
            raise TypeError("adapter callable must return str")
        return result


class LazyBackendAdapter:
    """Load an optional dLLM/AR adapter factory only on the first generation.

    A plugin path has the form ``package.module:factory``. The factory receives
    keyword configuration and must return an object implementing ModelAdapter.
    This package never imports torch, transformers, or a dLLM runtime itself.
    """

    def __init__(
        self,
        factory_path: str,
        *,
        backend_kind: str = "custom",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if ":" not in factory_path:
            raise ValueError("factory_path must have form 'module:attribute'")
        if backend_kind not in {"dllm", "ar", "custom"}:
            raise ValueError("backend_kind must be dllm, ar, or custom")
        self.factory_path = factory_path
        self.backend_kind = backend_kind
        self.config = dict(config or {})
        self._delegate: ModelAdapter | None = None

    @property
    def loaded(self) -> bool:
        return self._delegate is not None

    def generate(self, *, prompt: str, state: TaskState) -> str:
        delegate = self._load()
        return delegate.generate(prompt=prompt, state=state)

    def _load(self) -> ModelAdapter:
        if self._delegate is not None:
            return self._delegate
        module_name, attribute_path = self.factory_path.split(":", 1)
        if not module_name or not attribute_path:
            raise ValueError("factory_path must have non-empty module and attribute")
        target: Any = importlib.import_module(module_name)
        for part in attribute_path.split("."):
            target = getattr(target, part)
        delegate = target(**self.config)
        if not isinstance(delegate, ModelAdapter):
            raise TypeError("backend factory result does not implement ModelAdapter")
        self._delegate = delegate
        return delegate


class DemoAdapter:
    """Small rule adapter that demonstrates the loop without model weights."""

    _MATH = re.compile(r"^[\s\d.eE+*/%()\-]+$")

    def __init__(self, *, retrieval_available: bool = False) -> None:
        self.retrieval_available = retrieval_available

    def generate(self, *, prompt: str, state: TaskState) -> str:
        if state.history:
            observation = state.history[-1].observation
            if observation.error:
                answer = f"Tool failed safely: {observation.error}"
            else:
                answer = observation.summary
            return json.dumps({"type": "final", "answer": answer}, ensure_ascii=False)

        request = state.request.strip()
        if self._MATH.fullmatch(request):
            return json.dumps(
                {"type": "tool", "tool": "calculator", "arguments": {"expression": request}}
            )
        if self.retrieval_available:
            return json.dumps(
                {"type": "tool", "tool": "retrieve", "arguments": {"query": request, "top_k": 3}},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "type": "final",
                "answer": "Demo mode needs --docs for retrieval, or an arithmetic-only query.",
            }
        )
