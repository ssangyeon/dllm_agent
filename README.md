# dLLM Agent

This directory is a self-contained reference implementation of a bounded,
tool-using language-model loop. The core runtime uses only the Python standard
library. Optional local-model factories use the Torch and Transformers already
installed in the selected GPU environment. An autoregressive model, a diffusion
language model, or a deterministic replay implements the same three-line
`ModelAdapter` protocol.

The implementation is original. It does not vendor or copy source from the
repositories listed in [NOTICE](NOTICE).

## What is implemented

| Requirement | Implementation |
| --- | --- |
| Backend abstraction | `adapters.ModelAdapter` protocol |
| Optional dLLM/AR runtime | concrete local-snapshot factories loaded only on first `generate` |
| Explicit task state | `TaskState` transition methods and structured `StepRecord` history |
| Strict model actions | exact JSON schemas, duplicate-key rejection, finite numbers, depth/size limits |
| One targeted repair | one callback after the first parse failure; never a retry loop |
| Tools | registry, argument validation, normalized execution, canonical SHA-256 cache |
| Safe calculation | bounded AST interpreter; no `eval`, calls, names, attributes, or indexing |
| Local retrieval | deterministic Unicode keyword matching under one configured root |
| Observations | `summary`, `raw_reference`, `error`, and `reliability` on every tool path |
| Loop safety | max steps, repeated action de-duplication, repeated-observation/no-progress stop |
| Auditability | one append-only JSONL file per sanitized task ID, including event latency |
| Evaluation | replay plus live manifests; success, action validity, repair, observation, step, latency and termination metrics |
| Verification | stdlib `unittest` unit and integration suite with fake and replay adapters |

## Architecture

```text
ModelAdapter -> strict action parser -> ToolRegistry / ToolExecutor
     ^                                      |
     |                                      v
prompt snapshot <- explicit TaskState <- Observation
                           |
                           +-> per-task JSONL events / benchmark metrics
```

There is no hidden scratchpad or implicit mutation. Model output is either:

```json
{"type":"tool","tool":"calculator","arguments":{"expression":"2+2"}}
```

or:

```json
{"type":"final","answer":"The result is 4."}
```

No Markdown fences, unknown keys, duplicate JSON keys, `NaN`, or `Infinity`
are accepted. Parser input size, JSON depth, task/request size, tool catalog
metadata, and observation text all have explicit bounds. If parsing fails, the
engine asks the model once to repair only the syntax/schema. A second invalid
result terminates the task.

## Quick start

Python 3.10 or newer is sufficient; no runtime package needs to be installed.

```bash
cd dllm_agent
PYTHONPATH=src python3 -m dllm_final demo --query '2 + 3 * 4' --task-id quickstart
```

Search a fixed local document tree:

```bash
PYTHONPATH=src python3 -m dllm_final demo \
  --query 'explicit state update' \
  --docs examples/docs \
  --logs run-logs \
  --task-id retrieval-demo
```

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

An editable install is optional:

```bash
python3 -m pip install -e .
dllm-agent demo --query '144 / 12'
```

## Concrete local backends

Both factories accept only absolute local snapshot directories and set
`local_files_only=True`. Importing `dllm_final`, `optional_backends`, or running
the stdlib tests does not import Torch or Transformers. `LazyBackendAdapter`
loads a model once, on the first task call, and shares that loaded model across
the full live manifest.

### LLaDA-8B-Instruct

`dllm_final.optional_backends.llada:create_llada` loads a caller-supplied local
LLaDA snapshot with `AutoModel(..., trust_remote_code=True)`. Its sampling loop
was written for this project: it starts with a fixed mask canvas, makes greedy
proposals, ranks each masked position by normalized log probability, commits
the most confident quota, and leaves low-confidence positions masked for later
rounds. Position order breaks confidence ties, temperature is absent, and the
same inputs/settings therefore follow the same decoding path.

```bash
PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m dllm_final demo \
  --adapter dllm_final.optional_backends.llada:create_llada \
  --backend-kind dllm \
  --adapter-config '{"model_path":"/models/llada-8b-instruct","steps":64,"max_new_tokens":128,"trust_remote_code":true}' \
  --query 'Return a final action whose answer is OK.'
```

The selected snapshot may contain executable custom modeling code. The factory
refuses to load it unless `trust_remote_code=true` is supplied explicitly after
review. `local_files_only` prevents a network lookup but does not sandbox cached
remote code.

### Autoregressive Transformers baseline

`dllm_final.optional_backends.ar:create_ar_transformers` loads a local causal-LM
snapshot with `AutoModelForCausalLM` and performs greedy (`do_sample=False`)
generation. A compatible local instruct-model snapshot can exercise the same
parser, tools, state, termination rules, task manifest, and metrics as the dLLM
backend.

```bash
PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m dllm_final demo \
  --adapter dllm_final.optional_backends.ar:create_ar_transformers \
  --backend-kind ar \
  --adapter-config '{"model_path":"/models/llama-3.1-8b-instruct","max_new_tokens":128}' \
  --query 'Return a final action whose answer is OK.'
```

Both adapters return decoded model text unchanged apart from special-token/EOS
trimming and surrounding whitespace. They do not strip fences, extract JSON,
or synthesize an action; `StrictActionParser` and its single repair attempt
remain the only acceptance path.

## Local retrieval behavior

`KeywordRetriever` recursively reads `.txt`, `.md`, and `.rst` files below the
configured root. The model cannot supply or override a path, and symlinks are
ignored. Query characters/unique tokens, directory entries, directories, files,
per-file and total bytes, lines, and per-line characters have explicit limits.
Search retains only the requested top-k hits instead of accumulating every
match. Scoring is integer keyword overlap, and ties are ordered by relative
path and line number. Thus a fixed document tree and query produce the same
result.

This retriever is a transparent baseline, not semantic search. A host can
register a vector retriever behind `ToolSpec` while keeping the observation,
state, cache, logging, and termination contracts unchanged.

## Replay benchmark

`examples/benchmark.json` contains deterministic model outputs:

```bash
PYTHONPATH=src python3 -m dllm_final benchmark examples/benchmark.json
```

Each case has this schema:

```json
{
  "task_id": "calc-1",
  "request": "2 + 2",
  "outputs": [
    "{\"type\":\"tool\",\"tool\":\"calculator\",\"arguments\":{\"expression\":\"2+2\"}}",
    "{\"type\":\"final\",\"answer\":\"4\"}"
  ],
  "expected_contains": "4"
}
```

The summary reports task count, completion and expected-match rates, terminated
tasks, mean steps/latency, repair-task rate, duplicate-action rate, and tool
error rate. Latency is operational timing rather than a deterministic score.

## Live backend benchmark

`examples/live_manifest.json` contains one calculator task and one retrieval
task. A live case has `task_id`, `request`, optional case-insensitive
`expected_contains`, and optional unique `required_tools`. Run it through one
shared lazy backend:

```bash
PYTHONPATH=src python -m dllm_final live-benchmark examples/live_manifest.json \
  --adapter dllm_final.optional_backends.llada:create_llada \
  --backend-kind dllm \
  --adapter-config '{"model_path":"/models/llada-8b-instruct","trust_remote_code":true}' \
  --docs examples/docs \
  --logs run-logs
```

Task success requires completion, the expected answer substring (when given),
and the required observation flow (when tools are listed). The report also
includes completed/terminated tasks, total and mean
model calls, strict valid actions per model call, successful one-shot repairs
per repair attempt, mean steps/tool calls, tool errors, duplicate actions,
mean/median/nearest-rank-p95 end-to-end latency, and counts by termination
reason. Observation use is deliberately operational rather than a claim about
hidden reasoning: every `required_tools` observation must succeed, a later model
call must have received observation-bearing state, and the case must pass its
external answer check.

For a Slurm cluster, provide the local model path and choose the site's GPU
partition at submission time:

```bash
export LLADA_MODEL_PATH=/models/llada-8b-instruct
# Set this only after reviewing the snapshot's executable custom model code.
export TRUST_REMOTE_CODE=1
sbatch --partition=<gpu-partition> scripts/slurm_llada_live.sbatch

export LLAMA_MODEL_PATH=/models/llama-3.1-8b-instruct
sbatch --partition=<gpu-partition> scripts/slurm_ar_live.sbatch
```

Both scripts request one GPU, force offline Hugging Face behavior, and require
the model path explicitly. The LLaDA launcher additionally refuses to run until
`TRUST_REMOTE_CODE=1` records the operator's explicit opt-in. `PROJECT_ROOT`,
`PYTHON_BIN`, `MANIFEST`, and `DOCS_DIR` are overrideable. They create the live
JSON result and task-level JSONL logs under `results/` and `run-logs/`; Slurm's
own stdout/stderr files are written in the submission directory.

## JSONL events

Supplying `--logs DIRECTORY` creates one file per task. The filename contains a
sanitized task slug plus a SHA-256 prefix, while every row retains the exact
task ID. Rows contain UTC timestamp, event, current step, `latency_ms`, and a
structured payload. Model text itself is not logged; only character count,
parsed action, observation, and outcome are recorded. If observations can hold
sensitive material, apply the host project's redaction and retention policy.
Do not commit runtime logs, result files, credentials, private keys, local model
snapshots, or environment-specific paths. The supplied `.gitignore` excludes
the standard locations and model-file extensions used by this project.

## Grafting into an existing repository

1. Copy `src/dllm_final` under the host's package namespace.
2. Preserve the schemas in `actions.py` and transition methods in `types.py`.
3. Implement a local `ModelAdapter` factory; leave heavyweight imports lazy.
4. Register project tools with exact argument validators and `Observation`
   results. Mark nondeterministic tools `cacheable=False`.
5. Point retrieval at an explicit approved data directory.
6. Send `TaskJSONLLogger` to the host's protected log directory.
7. Port and run `tests`; add replay cases captured from the selected backend.

`AgentEngine` supplies a fresh cache mapping for every `run`, so its de-duplication
does not cross task runs. Direct callers of `ToolExecutor.execute` use the
executor's default cache unless they pass their own mapping; do not share that
default cache across unrelated users if observations may be sensitive. Also
treat model outputs and retrieved content as untrusted input even though the
parser and tools enforce their local contracts.

`max_steps` and `max_no_progress` bound the number of completed loop iterations;
they cannot preempt a backend or plugin that never returns. Production hosts
must enforce provider request timeouts and run untrusted or potentially hanging
tools behind a cancellable process/job boundary.

## Files

```text
src/dllm_final/
  actions.py      strict schema, canonical action hashes, one-shot repair
  adapters.py     protocol, replay/fake-friendly and lazy plugin adapters
  benchmark.py    replay cases and aggregate metrics
  cli.py          demo and benchmark commands
  engine.py       bounded observe-act-update loop
  eventlog.py     per-task JSONL audit writer
  optional_backends/ caller-supplied local LLaDA and AR adapters with lazy imports
  tools.py        registry/executor, AST calculator, keyword retrieval
  types.py        Observation, StepRecord, TaskState
tests/            stdlib-only unit and integration tests
examples/         local documents and replay benchmark input
scripts/          portable offline Slurm launchers
```

See [NOTICE](NOTICE) for framework/model provenance and
[REFERENCES.md](REFERENCES.md) for the pinned related-repository audit.
