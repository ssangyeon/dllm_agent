"""Strict action schema, validation, canonicalization, and one-shot repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Literal, Mapping


class ActionParseError(ValueError):
    pass


_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Action:
    kind: Literal["tool", "final"]
    tool: str | None = None
    arguments: Mapping[str, Any] | None = None
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "tool":
            return {"type": "tool", "tool": self.tool, "arguments": dict(self.arguments or {})}
        return {"type": "final", "answer": self.answer}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParseOutcome:
    action: Action
    repaired: bool
    raw: str


class StrictActionParser:
    """Parse only the documented JSON schema; never guess or strip Markdown."""

    schema = (
        'Either {"type":"tool","tool":"NAME","arguments":{...}} '
        'or {"type":"final","answer":"NON-EMPTY TEXT"}. No other keys.'
    )

    def __init__(self, *, max_chars: int = 16_384, max_depth: int = 12) -> None:
        if max_chars < 64 or max_depth < 1:
            raise ValueError("parser limits are too small")
        self.max_chars = max_chars
        self.max_depth = max_depth

    def parse(self, raw: str) -> Action:
        if not isinstance(raw, str):
            raise ActionParseError("model output must be text")
        if len(raw) > self.max_chars:
            raise ActionParseError(f"model output exceeds {self.max_chars} characters")
        if not raw.strip():
            raise ActionParseError("model output is empty")
        try:
            payload = json.loads(
                raw,
                object_pairs_hook=self._object_no_duplicates,
                parse_constant=self._reject_constant,
                parse_float=self._finite_float,
            )
        except ActionParseError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ActionParseError(f"invalid JSON: {exc}") from exc
        self._validate_json_tree(payload, depth=0)
        if not isinstance(payload, dict):
            raise ActionParseError("top-level JSON value must be an object")

        kind = payload.get("type")
        if kind == "tool":
            expected = {"type", "tool", "arguments"}
            self._require_exact_keys(payload, expected)
            tool = payload["tool"]
            arguments = payload["arguments"]
            if not isinstance(tool, str) or not _TOOL_NAME.fullmatch(tool):
                raise ActionParseError("tool must match [A-Za-z][A-Za-z0-9_.-]{0,63}")
            if not isinstance(arguments, dict):
                raise ActionParseError("arguments must be a JSON object")
            return Action(kind="tool", tool=tool, arguments=arguments)
        if kind == "final":
            expected = {"type", "answer"}
            self._require_exact_keys(payload, expected)
            answer = payload["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ActionParseError("answer must be a non-empty string")
            return Action(kind="final", answer=answer.strip())
        raise ActionParseError("type must be exactly 'tool' or 'final'")

    def parse_with_one_repair(
        self,
        raw: str,
        repair: Callable[[str], str],
    ) -> ParseOutcome:
        """Attempt a targeted model repair exactly once after an initial failure."""

        try:
            return ParseOutcome(action=self.parse(raw), repaired=False, raw=raw)
        except ActionParseError as first_error:
            prompt = self.repair_prompt(raw, first_error)
            repaired_raw = repair(prompt)  # deliberately one call, with no retry loop
            try:
                action = self.parse(repaired_raw)
            except ActionParseError as second_error:
                raise ActionParseError(
                    f"repair failed after one attempt; initial={first_error}; repaired={second_error}"
                ) from second_error
            return ParseOutcome(action=action, repaired=True, raw=repaired_raw)

    def repair_prompt(self, raw: str, error: Exception) -> str:
        bounded = raw[: min(len(raw), 4_000)]
        return (
            "Your previous action was invalid. Repair only its syntax/schema. "
            "Do not change the intended tool or answer. Return one JSON object and nothing else.\n"
            f"Schema: {self.schema}\nValidation error: {error}\nInvalid output: {bounded}"
        )

    @staticmethod
    def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActionParseError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> Any:
        raise ActionParseError(f"non-finite JSON number is forbidden: {value}")

    @staticmethod
    def _finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ActionParseError("non-finite JSON number is forbidden")
        return number

    @staticmethod
    def _require_exact_keys(payload: dict[str, Any], expected: set[str]) -> None:
        actual = set(payload)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ActionParseError(f"wrong keys; missing={missing}, extra={extra}")

    def _validate_json_tree(self, value: Any, *, depth: int) -> None:
        if depth > self.max_depth:
            raise ActionParseError(f"JSON nesting exceeds {self.max_depth}")
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ActionParseError("non-finite JSON number is forbidden")
            return
        if isinstance(value, list):
            for item in value:
                self._validate_json_tree(item, depth=depth + 1)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ActionParseError("object keys must be strings")
                self._validate_json_tree(item, depth=depth + 1)
            return
        raise ActionParseError(f"unsupported JSON value: {type(value).__name__}")
