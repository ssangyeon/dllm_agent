"""Tool registry, canonical execution cache, safe calculator, and retrieval."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from .actions import Action
from .types import Observation


class ToolError(ValueError):
    """Expected, user-visible tool validation/execution error."""


ToolHandler = Callable[[Mapping[str, Any]], Observation]
ArgumentValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler
    validate: ArgumentValidator
    argument_schema: str = "{}"
    cacheable: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", self.name):
            raise ValueError("tool name does not match the action schema")
        if not isinstance(self.description, str):
            raise ValueError("tool description must be a string")
        if len(self.description) > 512:
            raise ValueError("tool description exceeds 512 characters")
        if not self.description or self.description.isspace():
            raise ValueError("tool description must be a non-empty string")
        if not isinstance(self.argument_schema, str) or len(self.argument_schema) > 1_000:
            raise ValueError("tool argument_schema must be a string of at most 1000 characters")
        if not callable(self.handler) or not callable(self.validate):
            raise TypeError("tool handler and validator must be callable")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"tool name is empty or already registered: {spec.name!r}")
        if len(self._tools) >= 64:
            raise ValueError("registry cannot contain more than 64 tools")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def prompt_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "name": name,
                "description": self._tools[name].description,
                "arguments": self._tools[name].argument_schema,
            }
            for name in sorted(self._tools)
        ]


@dataclass(frozen=True)
class ExecutionResult:
    observation: Observation
    action_hash: str
    cached: bool
    latency_ms: float


class ToolExecutor:
    """Execute registered tools and de-duplicate canonical actions."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._cache: dict[str, Observation] = {}

    def execute(
        self,
        action: Action,
        *,
        cache: MutableMapping[str, Observation] | None = None,
    ) -> ExecutionResult:
        if action.kind != "tool" or action.tool is None or action.arguments is None:
            raise ValueError("ToolExecutor accepts only complete tool actions")
        cache_store = self._cache if cache is None else cache
        action_hash = action.canonical_hash()
        started = perf_counter()
        if action_hash in cache_store:
            return ExecutionResult(
                observation=cache_store[action_hash],
                action_hash=action_hash,
                cached=True,
                latency_ms=(perf_counter() - started) * 1_000,
            )

        cacheable = True  # unknown-tool and pre-dispatch failures are deterministic
        try:
            spec = self.registry.get(action.tool)
            cacheable = spec.cacheable
            # A tool receives a private copy so even a buggy plugin cannot mutate
            # the action after its canonical hash has been calculated.
            arguments = copy.deepcopy(dict(action.arguments))
            spec.validate(arguments)
            observation = spec.handler(arguments)
            if not isinstance(observation, Observation):
                raise TypeError("tool handler must return Observation")
        except Exception as exc:  # normalize plugin failures into the observation contract
            observation = Observation(
                summary=f"Tool '{action.tool}' failed safely.",
                raw_reference=action.canonical_json(),
                error=f"{type(exc).__name__}: {exc}",
                reliability=0.0,
            )

        if cacheable:
            cache_store[action_hash] = observation
        return ExecutionResult(
            observation=observation,
            action_hash=action_hash,
            cached=False,
            latency_ms=(perf_counter() - started) * 1_000,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


class SafeCalculator:
    """Evaluate arithmetic AST nodes without names, calls, attributes, or eval."""

    _BINARY: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    _UNARY: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }

    def __init__(
        self,
        *,
        max_chars: int = 512,
        max_nodes: int = 128,
        max_abs: float = 1e100,
        max_exponent: float = 12.0,
    ) -> None:
        self.max_chars = max_chars
        self.max_nodes = max_nodes
        self.max_abs = max_abs
        self.max_exponent = max_exponent

    def evaluate(self, expression: str) -> int | float:
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("expression must be a non-empty string")
        if len(expression) > self.max_chars:
            raise ToolError(f"expression exceeds {self.max_chars} characters")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"invalid arithmetic syntax: {exc.msg}") from exc
        if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
            raise ToolError(f"expression exceeds {self.max_nodes} AST nodes")
        value = self._visit(tree.body)
        return self._bounded(value)

    def _visit(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError("only integer and floating-point literals are allowed")
            return self._bounded(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY:
            return self._bounded(self._UNARY[type(node.op)](self._visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > self.max_exponent:
                raise ToolError(f"absolute exponent exceeds {self.max_exponent:g}")
            try:
                value = self._BINARY[type(node.op)](left, right)
            except (ArithmeticError, OverflowError, ValueError) as exc:
                raise ToolError(f"arithmetic error: {exc}") from exc
            if isinstance(value, complex):
                raise ToolError("complex results are not supported")
            return self._bounded(value)
        raise ToolError(f"forbidden syntax: {type(node).__name__}")

    def _bounded(self, value: int | float) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolError("result is not a real number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError("result is not finite")
        if abs(value) > self.max_abs:
            raise ToolError(f"absolute result exceeds {self.max_abs:g}")
        return value


def _require_exact_arguments(
    arguments: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(arguments)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        raise ToolError(f"wrong arguments; missing={sorted(missing)}, extra={sorted(extra)}")


def calculator_tool(calculator: SafeCalculator | None = None) -> ToolSpec:
    calculator = calculator or SafeCalculator()

    def validate(arguments: Mapping[str, Any]) -> None:
        _require_exact_arguments(arguments, required={"expression"})
        if not isinstance(arguments["expression"], str):
            raise ToolError("expression must be a string")

    def handle(arguments: Mapping[str, Any]) -> Observation:
        expression = str(arguments["expression"])
        value = calculator.evaluate(expression)
        return Observation(
            summary=f"Calculation result: {value!r}",
            raw_reference=expression,
            error=None,
            reliability=1.0,
        )

    return ToolSpec(
        name="calculator",
        description="Evaluate bounded arithmetic using a safe Python AST subset.",
        handler=handle,
        validate=validate,
        argument_schema='{"expression":"non-empty arithmetic string"}',
    )


@dataclass(frozen=True)
class SearchHit:
    score: int
    relative_path: str
    line_number: int
    snippet: str


class KeywordRetriever:
    """Deterministic keyword-overlap retrieval over a fixed local root."""

    _TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

    def __init__(
        self,
        root: str | Path,
        *,
        extensions: Sequence[str] = (".txt", ".md", ".rst"),
        max_file_bytes: int = 1_000_000,
        max_total_bytes: int = 10_000_000,
        max_files: int = 1_000,
        max_directories: int = 1_000,
        max_entries: int = 20_000,
        max_lines: int = 100_000,
        max_line_chars: int = 16_000,
        max_query_chars: int = 2_000,
        max_query_tokens: int = 64,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"document root is not a directory: {self.root}")
        self.extensions = frozenset(ext.casefold() for ext in extensions)
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_files = max_files
        self.max_directories = max_directories
        self.max_entries = max_entries
        self.max_lines = max_lines
        self.max_line_chars = max_line_chars
        self.max_query_chars = max_query_chars
        self.max_query_tokens = max_query_tokens
        limits = (
            max_file_bytes,
            max_total_bytes,
            max_files,
            max_directories,
            max_entries,
            max_lines,
            max_line_chars,
            max_query_chars,
            max_query_tokens,
        )
        if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 for limit in limits):
            raise ValueError("retrieval resource limits must be positive integers")

    @classmethod
    def tokens(cls, text: str) -> list[str]:
        return cls._TOKEN.findall(text.casefold())

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if not isinstance(query, str):
            raise ToolError("query must be a string")
        if len(query) > self.max_query_chars:
            raise ToolError(f"query exceeds {self.max_query_chars} characters")
        if not query or query.isspace():
            raise ToolError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ToolError("top_k must be an integer between 1 and 10")
        query_counts = Counter(self.tokens(query))
        if not query_counts:
            raise ToolError("query contains no searchable tokens")
        if len(query_counts) > self.max_query_tokens:
            raise ToolError(f"query exceeds {self.max_query_tokens} unique tokens")

        hits: list[SearchHit] = []
        total_bytes = 0
        total_lines = 0
        normalized_query = query.casefold().strip()
        for path in self._document_paths():
            try:
                size = path.stat().st_size
                if size > self.max_file_bytes:
                    continue
                total_bytes += size
                if total_bytes > self.max_total_bytes:
                    raise ToolError(
                        f"document bytes exceed configured limit {self.max_total_bytes}"
                    )
                handle = path.open("r", encoding="utf-8", errors="replace")
            except OSError:
                continue
            relative = path.relative_to(self.root).as_posix()
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    total_lines += 1
                    if total_lines > self.max_lines:
                        raise ToolError(
                            f"document lines exceed configured limit {self.max_lines}"
                        )
                    bounded_line = line[: self.max_line_chars]
                    line_counts = Counter(self.tokens(bounded_line))
                    overlap = sum(
                        min(count, line_counts[token]) for token, count in query_counts.items()
                    )
                    if not overlap:
                        continue
                    unique_overlap = sum(1 for token in query_counts if line_counts[token])
                    phrase_bonus = 2 if normalized_query in bounded_line.casefold() else 0
                    score = overlap * 10 + unique_overlap + phrase_bonus
                    snippet = " ".join(bounded_line.strip().split())[:240]
                    hits.append(SearchHit(score, relative, line_number, snippet))
                    hits.sort(key=self._hit_rank)
                    if len(hits) > top_k:
                        hits.pop()
        return hits

    @staticmethod
    def _hit_rank(hit: SearchHit) -> tuple[int, str, str, int]:
        return (-hit.score, hit.relative_path.casefold(), hit.relative_path, hit.line_number)

    def _document_paths(self) -> list[Path]:
        """Walk deterministically without following symlinks or unbounded rglob lists."""

        paths: list[Path] = []
        pending = [self.root]
        directories = 0
        entries_seen = 0
        while pending:
            current = pending.pop()
            directories += 1
            if directories > self.max_directories:
                raise ToolError(
                    f"document directories exceed configured limit {self.max_directories}"
                )
            try:
                with os.scandir(current) as iterator:
                    entries: list[os.DirEntry[str]] = []
                    for entry in iterator:
                        entries_seen += 1
                        if entries_seen > self.max_entries:
                            raise ToolError(
                                f"document entries exceed configured limit {self.max_entries}"
                            )
                        entries.append(entry)
            except OSError:
                continue
            entries.sort(key=lambda entry: (entry.name.casefold(), entry.name))
            subdirectories: list[Path] = []
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        subdirectories.append(Path(entry.path))
                    elif (
                        entry.is_file(follow_symlinks=False)
                        and Path(entry.name).suffix.casefold() in self.extensions
                    ):
                        paths.append(Path(entry.path))
                        if len(paths) > self.max_files:
                            raise ToolError(
                                f"document files exceed configured limit {self.max_files}"
                            )
                except OSError:
                    continue
            # Stack is LIFO; reverse insertion keeps traversal alphabetical.
            pending.extend(reversed(subdirectories))
        return sorted(paths, key=lambda item: self._path_rank(item))

    def _path_rank(self, item: Path) -> tuple[str, str]:
        relative = item.relative_to(self.root).as_posix()
        return (relative.casefold(), relative)


def retrieval_tool(retriever: KeywordRetriever) -> ToolSpec:
    def validate(arguments: Mapping[str, Any]) -> None:
        _require_exact_arguments(arguments, required={"query"}, optional={"top_k"})
        if not isinstance(arguments["query"], str):
            raise ToolError("query must be a string")
        top_k = arguments.get("top_k", 3)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolError("top_k must be an integer")

    def handle(arguments: Mapping[str, Any]) -> Observation:
        query = str(arguments["query"])
        top_k = int(arguments.get("top_k", 3))
        hits = retriever.search(query, top_k=top_k)
        references = [
            {
                "path": hit.relative_path,
                "line": hit.line_number,
                "score": hit.score,
            }
            for hit in hits
        ]
        if not hits:
            return Observation(
                summary="No keyword matches found in the configured local documents.",
                raw_reference=json.dumps(
                    {"root": str(retriever.root), "hits": []},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                error=None,
                reliability=0.2,
            )
        summary = "\n".join(
            f"[{hit.relative_path}:{hit.line_number}] {hit.snippet}" for hit in hits
        )
        return Observation(
            summary=summary,
            raw_reference=json.dumps(references, ensure_ascii=False, sort_keys=True),
            error=None,
            reliability=min(1.0, 0.55 + 0.1 * len(hits)),
        )

    return ToolSpec(
        name="retrieve",
        description="Search UTF-8 .txt/.md/.rst files below the configured local document root.",
        handler=handle,
        validate=validate,
        argument_schema='{"query":"non-empty string","top_k":"optional integer 1..10"}',
    )


def default_registry(*, documents: str | Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator_tool())
    if documents is not None:
        registry.register(retrieval_tool(KeywordRetriever(documents)))
    return registry
