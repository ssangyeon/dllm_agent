"""Greedy Transformers adapter for a caller-supplied local model snapshot."""

from __future__ import annotations

from typing import Any

from ._common import (
    checked_snapshot_path,
    context_limit,
    decode_generated,
    positive_int,
    render_chat_prompt,
    stop_token_ids,
    torch_dtype,
)


class ARTransformersAdapter:
    def __init__(
        self,
        *,
        torch_module: Any,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
    ) -> None:
        self._torch = torch_module
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._stops = stop_token_ids(tokenizer, model)

    def generate(self, *, prompt: str, state: object) -> str:
        del state
        rendered = render_chat_prompt(self._tokenizer, prompt)
        encoded = self._tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        prompt_length = int(input_ids.shape[1])
        limit = context_limit(self._model, self._tokenizer)
        wanted = prompt_length + self._max_new_tokens
        if limit is not None and wanted > limit:
            raise ValueError(f"prompt plus generation is {wanted} tokens; model limit is {limit}")
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "eos_token_id", None)

        with self._torch.inference_mode():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
                pad_token_id=pad_token_id,
                use_cache=True,
            )
        generated = output[0, prompt_length:].tolist()
        return decode_generated(self._tokenizer, generated, self._stops)


def create_ar_transformers(
    *,
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 128,
    seed: int = 42,
    trust_remote_code: bool = False,
) -> ARTransformersAdapter:
    snapshot = checked_snapshot_path(model_path)
    positive_int("max_new_tokens", max_new_tokens)
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(trust_remote_code, bool):
        raise TypeError("trust_remote_code must be a boolean")

    # Heavy imports are intentionally inside this lazy factory.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected_dtype = torch_dtype(torch, dtype)
    if device.startswith("cuda") and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested on a GPU without BF16 support")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=selected_dtype,
    ).to(device).eval()
    return ARTransformersAdapter(
        torch_module=torch,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
    )
