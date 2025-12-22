#!/bin/bash
set -euo pipefail
# Usage: bash scripts/run_mol_edit_eval_bench.sh [MODEL_DIR] [BENCH_DIR] [OUT_DIR]
# Defaults point to your 7B ckpt and ChemCoTBench benchmark.


MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/moledit-7b/}
# MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/model/Qwen2.5-7B-Instruct}
BENCH_DIR=${2:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/ChemCoTBench_benchmark/chemcotbench/mol_edit}
OUT_DIR=${3:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/mol_edit_eval_bench_7b_maxturn_16/}
MAX_NEW=${MAX_NEW_TOKENS:-2048}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}

mkdir -p "$OUT_DIR"

# Use async_verl backend by default to match training setup
# If you encounter issues, try: BACKEND=transformers (simpler, slower, no multiprocessing issues)
BACKEND=${BACKEND:-async_verl}

for split in add delete sub; do
  python scripts/run_mol_edit_agent.py \
    --model_path "$MODEL_DIR" \
    --input_file "$BENCH_DIR/${split}.json" \
    --output_file "$OUT_DIR/pred_${split}.json" \
    --max_new_tokens $MAX_NEW \
    --temperature $TEMP \
    --top_p $TOP_P \
    --backend "$BACKEND"
  echo "saved $OUT_DIR/pred_${split}.json"
done

echo "All done. Logs in $OUT_DIR"
