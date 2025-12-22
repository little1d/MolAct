#!/bin/bash
set -euo pipefail
# Usage: bash scripts/3 run_mol_opt_inference.sh [MODEL_DIR] [BENCH_DIR] [OUT_DIR]
# Defaults point to your 7B ckpt and ChemCoTBench benchmark.

# mol_opt-7b after training
# MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/mol_opt-7b}MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/checkpoints/AgentRL/mol_opt-7b}
MODEL_DIR=${1:-/mnt/shared-storage-user/yangzhuo/main/model/Qwen2.5-3B-Instruct}
BENCH_DIR=${2:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/ChemCoTBench_benchmark/chemcotbench/mol_opt}
OUT_DIR=${3:-/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/outputs/mol_opt_eval_bench_qwen_2.5_3b/}
MAX_NEW=${MAX_NEW_TOKENS:-2048}
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.95}

mkdir -p "$OUT_DIR"

# Set TDC cache path for oracle tools
export TDC_CACHE_PATH="/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/oracle"
export PYTDC_CACHE="$TDC_CACHE_PATH"

# Process each subtask separately
for subtask in logp drd jnk gsk qed solubility; do
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

