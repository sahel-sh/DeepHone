#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RETRIEVAL_OUTPUT_DIR="${1:-${PROJECT_ROOT}/retrieval_output}"
AGGREGATE_SCRIPT="${PROJECT_ROOT}/scripts_rerank/aggregate_token_stats_per_experiment.py"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-browsecomp}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH." >&2
  exit 1
fi

if [[ ! -d "${RETRIEVAL_OUTPUT_DIR}" ]]; then
  echo "Error: retrieval output directory not found: ${RETRIEVAL_OUTPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
  echo "Error: aggregate script not found: ${AGGREGATE_SCRIPT}" >&2
  exit 1
fi

PY_CMD=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)

echo "Using conda environment: ${CONDA_ENV_NAME}"
echo "Scanning run folders under: ${RETRIEVAL_OUTPUT_DIR}"

processed=0
shopt -s nullglob
for run_dir in "${RETRIEVAL_OUTPUT_DIR}"/*; do
  [[ -d "${run_dir}" ]] || continue
  echo "Aggregating token stats in: ${run_dir}"
  "${PY_CMD[@]}" "${AGGREGATE_SCRIPT}" "${run_dir}" --recursive
  processed=$((processed + 1))
done

echo "Done. Processed ${processed} run folder(s)."
