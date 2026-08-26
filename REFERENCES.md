# Related repository audit

The implementation in this repository is original. The projects below were
inspected to understand dLLM sampling, tool-agent loop design, and evaluation
interfaces. No source was copied from them. Commits are pinned so the review is
reproducible even if a default branch moves.

| Repository | Pinned commit | License finding | How it informed this project |
| --- | --- | --- | --- |
| [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) | `ca176752fbceec49c6b4777a2c18ae88e4eb10ed` | Apache-2.0 | Compared its unified sampler interface and LLaDA/Dream examples. Its package/import stack is not a runtime dependency here. |
| [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` | Apache-2.0 | Reviewed acceleration-oriented diffusion decoding as future optimization context; no kernel or sampler code was reused. |
| [RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | `ad86a6692be0d7fc4445c77938c090349815e997` | MIT | Used only as background for iterative search/tool interaction and external-observation loops. |
| [bubble65/DLLM-Searcher](https://github.com/bubble65/DLLM-Searcher) | `9deca06cb3da63a758dba4c6ffb18a156d47d6ab` | No top-level license found during review | Treated as concept-only and unavailable for code reuse. |
| [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | `9182493720ed723ef8031210d85959364e51cbe0` | No top-level license found during review | Used to identify the model family and checkpoint contract only. Repository generation code was not copied. |

The optional LLaDA adapter executes custom model code that is already present
inside a separately cached Hugging Face snapshot via `trust_remote_code=True`.
That checkpoint code and the weights are outside this repository and need their
own license and security review. The adapter's confidence-ranked mask-denoising
loop was written independently in `src/dllm_final/optional_backends/llada.py`.

See `NOTICE` for framework/library provenance and model-license boundaries.
