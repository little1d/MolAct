#!/bin/bash
set -euo pipefail
# Usage: bash scripts/run_mol_edit_eval_bench.sh <MODEL_DIR> <BENCH_DIR> <OUT_JSON>
# BENCH_DIR contains add.json/delete.json/sub.json (ChemCoTBench benchmark)

MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/moledit-7b/}
BENCH_DIR=${2:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/ChemCoTBench_benchmark/chemcotbench/mol_edit}
OUT_DIR=${3:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/mol_edit_eval_bench_7b/}
MAX_NEW=${MAX_NEW_TOKENS:-128}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}
BATCH=${BATCH:-1}

mkdir -p "$OUT_DIR"

for split in add delete sub; do
  python scripts/infer_mol_edit_bench.py \
    --model_path "$MODEL_DIR" \
    --input_file "$BENCH_DIR/${split}.json" \
    --output_file "$OUT_DIR/pred_${split}.json" \
    --max_new_tokens $MAX_NEW \
    --temperature $TEMP \
    --top_p $TOP_P \
    --batch_size $BATCH
  echo "saved $OUT_DIR/pred_${split}.json"
done

echo "All done. Logs in $OUT_DIR"
