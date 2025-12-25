#!/bin/bash
set -euo pipefail

# mol_opt-7b after training
MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/molopt-7b}
BENCH_DIR=${2:-/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/ChemCoTBench_benchmark/chemcotbench/mol_opt}
OUT_DIR=${3:-/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/mol_opt_eval_bench_7b_maxturn_16/}
MAX_NEW=${MAX_NEW_TOKENS:-2048}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}

mkdir -p "$OUT_DIR"

# Process each subtask separately
# for subtask in logp drd jnk gsk qed solubility; do
for subtask in logp; do
  echo "=========================================="
  echo "Processing subtask: $subtask"
  echo "=========================================="
  
  python scripts/run_mol_opt_agent.py \
    --model_path "$MODEL_DIR" \
    --input_file "$BENCH_DIR/${subtask}.json" \
    --output_file "$OUT_DIR/pred_${subtask}.json" \
    --max_new_tokens $MAX_NEW \
    --temperature $TEMP \
    --top_p $TOP_P
  
  echo "✓ Saved $OUT_DIR/pred_${subtask}.json"
  echo ""
done

echo "All done. Logs in $OUT_DIR"

