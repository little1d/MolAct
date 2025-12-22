#!/usr/bin/env bash

# Usage:
#   bash scripts/eval_mol_edit_bench.sh \
#     --bench_dir /path/to/ChemCoTBench/baseline_and_eval \
#     --pred_dir /path/to/outputs/mol_edit_eval_bench_7b \
#     --out_dir /path/to/outputs/chemcotbench_results_7b
#
# Defaults are set for your current workspace.

set -euo pipefail

# Set NLTK data path to avoid network requests and SSL errors
export NLTK_DATA=/mnt/shared-storage-user/yangzhuo/main/datasets/nltk_data
mkdir -p "${NLTK_DATA}"

BENCH_DIR="${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/ChemCoTBench/baseline_and_eval}"
# PRED_DIR="${2:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/mol_edit_eval_bench_qwen2.5_7b}"
PRED_DIR="${2:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/mol_edit_eval_bench_7b_maxturn_16/}"
OUT_DIR="${3:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/chemcotbench_moledit_results_moledit_7b_maxturn_16}"

mkdir -p "${OUT_DIR}"

# Use chemcot_eval environment Python if available, otherwise use system python
PYTHON_CMD="${PYTHON_CMD:-/mnt/shared-storage-user/yangzhuo/miniconda3/envs/tests/bin/python}"
if [ ! -f "${PYTHON_CMD}" ]; then
    PYTHON_CMD="python3"
fi

echo "Benchmark dir : ${BENCH_DIR}"
echo "Pred dir      : ${PRED_DIR}"
echo "Out dir       : ${OUT_DIR}"

# Evaluate add/delete/sub separately
for SUB in add delete sub; do
  PRED_FILE="${PRED_DIR}/pred_${SUB}.json"
  OUT_FILE="${OUT_DIR}/eval_${SUB}.json"

  if [ ! -f "${PRED_FILE}" ]; then
    echo "Missing prediction file: ${PRED_FILE}, skip..."
    continue
  fi

  echo "Evaluating ${SUB}..."
  "${PYTHON_CMD}" "${BENCH_DIR}/eval_moledit_cli.py" \
    --task mol_edit \
    --subtask "${SUB}" \
    --pred "${PRED_FILE}" \
    --out "${OUT_FILE}"
done

echo "Done. Results saved in ${OUT_DIR}"

