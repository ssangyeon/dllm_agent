#!/usr/bin/env bash
#SBATCH --job-name=dllm-agent-llada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
: "${LLADA_MODEL_PATH:?Set LLADA_MODEL_PATH to an absolute local snapshot directory}"
MODEL_PATH="${LLADA_MODEL_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/examples/live_manifest.json}"
DOCS_DIR="${DOCS_DIR:-${PROJECT_ROOT}/examples/docs}"

if [[ "${MODEL_PATH}" != /* || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "LLADA_MODEL_PATH must be an absolute local snapshot containing config.json" >&2
  exit 2
fi
if [[ "${TRUST_REMOTE_CODE:-}" != "1" ]]; then
  echo "Review the local snapshot, then set TRUST_REMOTE_CODE=1 to allow its custom code" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ADAPTER_CONFIG="$("${PYTHON_BIN}" -c 'import json, sys; print(json.dumps({"model_path": sys.argv[1], "device": "cuda", "dtype": "bfloat16", "steps": 32, "max_new_tokens": 128, "trust_remote_code": True}))' "${MODEL_PATH}")"
RUN_ID="${RUN_ID:-llada-${SLURM_JOB_ID:-manual}}"
LOG_DIR="run-logs/${RUN_ID}"
RESULT_PATH="results/${RUN_ID}.json"
mkdir -p "${LOG_DIR}" results

"${PYTHON_BIN}" -m dllm_final live-benchmark "${MANIFEST}" \
  --adapter dllm_final.optional_backends.llada:create_llada \
  --backend-kind dllm \
  --adapter-config "${ADAPTER_CONFIG}" \
  --docs "${DOCS_DIR}" \
  --logs "${LOG_DIR}" \
  --max-steps 6 \
  --max-no-progress 2 | tee "${RESULT_PATH}"
