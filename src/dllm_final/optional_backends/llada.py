"""Original deterministic low-confidence denoising adapter for LLaDA-8B.

This implementation uses only the checkpoint's Hugging Face remote model class;
it does not import or copy the upstream LLaDA generation script. At each round,
greedy token proposals are scored by normalized log probability. The most
confident proposals are committed and lower-confidence positions stay masked
for a later round.
"""

from __future__ import annotations

from math import isfinite
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


def _commit_count(masked: int, rounds_left: int) -> int:
    """Evenly exhaust masked positions while leaving uncertain ones for later."""

    positive_int("masked", masked)
    positive_int("rounds_left", rounds_left)
    return max(1, (masked + rounds_left - 1) // rounds_left)


def _confidence_order(positions: list[int], confidences: list[float]) -> list[int]:
    if len(positions) != len(confidences):
        raise ValueError("positions and confidences must have equal lengths")
    if not all(isfinite(value) for value in confidences):
        raise RuntimeError("model produced a non-finite confidence")
    # Position is an explicit tie-breaker, so equal model scores do not depend
    # on device-specific top-k tie ordering.
    return sorted(
        range(len(positions)),
        key=lambda index: (-confidences[index], positions[index]),
    )


class LLaDADeterministicAdapter:
    def __init__(
        self,
        *,
        torch_module: Any,
        model: Any,
        tokenizer: Any,
        device: str,
        mask_token_id: int,
        steps: int,
        max_new_tokens: int,
    ) -> None:
        self._torch = torch_module
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._mask_token_id = mask_token_id
        self._steps = steps
        self._max_new_tokens = max_new_tokens
        self._stops = stop_token_ids(tokenizer, model)

    def generate(self, *, prompt: str, state: object) -> str:
        del state
        torch = self._torch
        rendered = render_chat_prompt(self._tokenizer, prompt)
        encoded = self._tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("LLaDA adapter currently requires one unpadded prompt")
        prompt_length = int(input_ids.shape[1])
        limit = context_limit(self._model, self._tokenizer)
        wanted = prompt_length + self._max_new_tokens
        if limit is not None and wanted > limit:
            raise ValueError(f"prompt plus generation is {wanted} tokens; model limit is {limit}")

        mask_canvas = torch.full(
            (1, self._max_new_tokens),
            self._mask_token_id,
            dtype=torch.long,
            device=self._device,
        )
        tokens = torch.cat((input_ids, mask_canvas), dim=1)
        attention_mask = torch.ones_like(tokens)

        with torch.inference_mode():
            for round_index in range(self._steps):
                relative = torch.nonzero(
                    tokens[0, prompt_length:] == self._mask_token_id,
                    as_tuple=False,
                ).flatten()
                masked = int(relative.numel())
                if masked == 0:
                    break
                positions = relative + prompt_length
                outputs = self._model(input_ids=tokens, attention_mask=attention_mask)
                candidate_logits = outputs.logits[0, positions, :]
                proposals = torch.argmax(candidate_logits, dim=-1)
                proposal_logits = candidate_logits.gather(1, proposals.unsqueeze(1)).squeeze(1)
                confidences = proposal_logits - torch.logsumexp(candidate_logits, dim=-1)

                rounds_left = self._steps - round_index
                count = min(masked, _commit_count(masked, rounds_left))
                position_list = [int(item) for item in positions.tolist()]
                confidence_list = [float(item) for item in confidences.float().tolist()]
                order = _confidence_order(position_list, confidence_list)[:count]
                chosen = torch.tensor(order, dtype=torch.long, device=self._device)
                tokens[0, positions[chosen]] = proposals[chosen]

        remaining = int((tokens[0, prompt_length:] == self._mask_token_id).sum().item())
        if remaining:
            raise RuntimeError(f"denoising ended with {remaining} masked positions")
        generated = tokens[0, prompt_length:].tolist()
        return decode_generated(self._tokenizer, generated, self._stops)


def create_llada(
    *,
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    mask_token_id: int = 126336,
    steps: int = 64,
    max_new_tokens: int = 128,
    seed: int = 42,
    trust_remote_code: bool = False,
) -> LLaDADeterministicAdapter:
    """Load a caller-supplied local snapshot; network identifiers are rejected."""

    snapshot = checked_snapshot_path(model_path)
    positive_int("mask_token_id", mask_token_id)
    positive_int("steps", steps)
    positive_int("max_new_tokens", max_new_tokens)
    if steps > max_new_tokens:
        raise ValueError("steps cannot exceed max_new_tokens")
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(trust_remote_code, bool):
        raise TypeError("trust_remote_code must be a boolean")
    if not trust_remote_code:
        raise ValueError(
            "LLaDA requires custom model code; review the local snapshot and set "
            "trust_remote_code=true to continue"
        )

    # Heavy imports are intentionally inside this lazy factory.
    import torch
    from transformers import AutoModel, AutoTokenizer

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
    model = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=selected_dtype,
    ).to(device).eval()
    configured_mask = getattr(getattr(model, "config", None), "mask_token_id", None)
    tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
    for label, value in (("model", configured_mask), ("tokenizer", tokenizer_mask)):
        if value is not None and int(value) != mask_token_id:
            raise RuntimeError(f"{label} mask token {value} != configured {mask_token_id}")
    if getattr(tokenizer, "pad_token_id", None) == mask_token_id:
        raise RuntimeError("padding token and diffusion mask token must be different")
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vocab_size, int) and mask_token_id >= vocab_size:
        raise RuntimeError("diffusion mask token is outside the model vocabulary")
    return LLaDADeterministicAdapter(
        torch_module=torch,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mask_token_id=mask_token_id,
        steps=steps,
        max_new_tokens=max_new_tokens,
    )
