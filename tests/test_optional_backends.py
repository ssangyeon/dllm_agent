from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dllm_final.optional_backends._common import (
    context_limit,
    decode_generated,
    stop_token_ids,
    truncate_at_stop,
)
from dllm_final.optional_backends.ar import create_ar_transformers
from dllm_final.optional_backends.llada import (
    _commit_count,
    _confidence_order,
    create_llada,
)


class OptionalBackendTests(unittest.TestCase):
    def test_factories_require_an_explicit_model_path(self) -> None:
        with self.assertRaisesRegex(TypeError, "model_path"):
            create_llada()  # type: ignore[call-arg]
        with self.assertRaisesRegex(TypeError, "model_path"):
            create_ar_transformers()  # type: ignore[call-arg]

    def test_import_does_not_import_heavy_dependencies(self) -> None:
        code = (
            "import sys; import dllm_final.optional_backends; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_factories_reject_nonlocal_path_before_heavy_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute local"):
            create_llada(model_path="relative/model")
        with self.assertRaisesRegex(ValueError, "absolute local"):
            create_ar_transformers(model_path="relative/model")

    def test_factories_require_snapshot_config_before_heavy_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                create_llada(model_path=directory)
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                create_ar_transformers(model_path=directory)

    def test_llada_requires_explicit_custom_code_consent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trust_remote_code=true"):
                create_llada(model_path=directory)
            with self.assertRaisesRegex(TypeError, "trust_remote_code"):
                create_llada(model_path=directory, trust_remote_code="yes")  # type: ignore[arg-type]

    def test_low_confidence_schedule_exhausts_masks(self) -> None:
        remaining = 7
        rounds = 4
        committed: list[int] = []
        for index in range(rounds):
            count = min(remaining, _commit_count(remaining, rounds - index))
            committed.append(count)
            remaining -= count
        self.assertEqual(committed, [2, 2, 2, 1])
        self.assertEqual(remaining, 0)

    def test_confidence_order_has_position_tie_breaker(self) -> None:
        order = _confidence_order([8, 3, 5], [-0.2, -0.2, -0.1])
        self.assertEqual(order, [2, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _confidence_order([1], [float("nan")])

    def test_stop_trimming_and_decode_are_deterministic(self) -> None:
        class Tokenizer:
            eos_token_id = 9
            unk_token_id = -1

            @staticmethod
            def convert_tokens_to_ids(token: str) -> int:
                return {"<|eot_id|>": 7, "<|im_end|>": -1}[token]

            @staticmethod
            def decode(tokens: list[int], **kwargs: object) -> str:
                del kwargs
                return ",".join(map(str, tokens))

        class Model:
            class config:
                eos_token_id = [9, 11]

        stops = stop_token_ids(Tokenizer(), Model())
        self.assertEqual(stops, {7, 9, 11})
        self.assertEqual(truncate_at_stop([1, 2, 7, 3], stops), [1, 2])
        self.assertEqual(decode_generated(Tokenizer(), [1, 2, 9, 3], stops), "1,2")

    def test_context_limit_ignores_tokenizer_infinity_sentinel(self) -> None:
        class Model:
            class config:
                max_sequence_length = 4096
                max_position_embeddings = 8192

        class Tokenizer:
            model_max_length = 10**30

        self.assertEqual(context_limit(Model(), Tokenizer()), 4096)


if __name__ == "__main__":
    unittest.main()
