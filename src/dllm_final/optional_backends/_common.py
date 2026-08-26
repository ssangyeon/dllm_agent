"""Small dependency-free helpers shared by optional model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def checked_snapshot_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("model_path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("model_path must be an absolute local snapshot path")
    if not path.is_dir():
        raise FileNotFoundError(f"local model snapshot does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot has no config.json: {path}")
    return str(path)


def positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def torch_dtype(torch_module: Any, name: str) -> Any:
    supported = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    try:
        return supported[name]
    except (KeyError, TypeError) as exc:
        raise ValueError("dtype must be bfloat16, float16, or float32") from exc


def context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = (
        getattr(getattr(model, "config", None), "max_sequence_length", None),
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    finite = [
        int(value)
        for value in candidates
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000
    ]
    return min(finite) if finite else None


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text")
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise RuntimeError("tokenizer has no chat template; an instruct snapshot is required")
    rendered = template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("tokenizer chat template returned no text")
    return rendered


def stop_token_ids(tokenizer: Any, model: Any, extra: Iterable[int] = ()) -> set[int]:
    result: set[int] = {int(item) for item in extra if isinstance(item, int) and item >= 0}
    for source in (tokenizer, getattr(model, "config", None)):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int) and value >= 0:
            result.add(value)
        elif isinstance(value, (tuple, list)):
            result.update(int(item) for item in value if isinstance(item, int) and item >= 0)
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        unknown = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|eot_id|>", "<|im_end|>"):
            value = converter(token)
            if isinstance(value, int) and value >= 0 and value != unknown:
                result.add(value)
    return result


def truncate_at_stop(token_ids: Iterable[int], stops: set[int]) -> list[int]:
    result: list[int] = []
    for value in token_ids:
        token_id = int(value)
        if token_id in stops:
            break
        result.append(token_id)
    return result


def decode_generated(tokenizer: Any, token_ids: Iterable[int], stops: set[int]) -> str:
    trimmed = truncate_at_stop(token_ids, stops)
    try:
        text = tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode(trimmed, skip_special_tokens=True)
    if not isinstance(text, str):
        raise TypeError("tokenizer.decode must return text")
    return text.strip()
