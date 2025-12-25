#!/usr/bin/env bash

# Usage:
#   bash scripts/4 eval_mol_opt_chemcotbench.sh \
#     [BENCH_DIR] [PRED_DIR] [OUT_DIR]
#
# Defaults are set for your current workspace.
#
# This script evaluates molecular optimization predictions using ChemCoTBench evaluator.
# It processes each subtask (logp, drd, jnk, gsk, qed, solubility) separately.

set -euo pipefail

# Set NLTK data path to avoid network requests and SSL errors
export NLTK_DATA=/mnt/shared-storage-user/yangzhuo/main/datasets/nltk_data
mkdir -p "${NLTK_DATA}"

# 注意这里的 BENCH_DIR是 ChemCoTBench的baseline_and_eval 目录，和inference的BENCH_DIR不同
BENCH_DIR="${1:-/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/ChemCoTBench/baseline_and_eval}"
PRED_DIR="${2:-/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/mol_opt_eval_bench_7b_maxturn_16/}"
OUT_DIR="${3:-/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/chemcotbench_molopt_results_7b_maxturn_16}"

mkdir -p "${OUT_DIR}"

# Use tests environment Python if available, otherwise use system python
PYTHON_CMD="${PYTHON_CMD:-/mnt/shared-storage-user/yangzhuo/miniconda3/envs/tests/bin/python}"
if [ ! -f "${PYTHON_CMD}" ]; then
    PYTHON_CMD="python3"
fi

echo "Benchmark dir : ${BENCH_DIR}"
echo "Pred dir      : ${PRED_DIR}"
echo "Out dir       : ${OUT_DIR}"
echo "Python        : ${PYTHON_CMD}"

# Evaluate each subtask separately
# for SUBTASK in logp drd jnk gsk qed solubility; do
for SUBTASK in logp; do
  PRED_FILE="${PRED_DIR}/pred_${SUBTASK}.json"
  OUT_FILE="${OUT_DIR}/eval_${SUBTASK}.json"

  if [ ! -f "${PRED_FILE}" ]; then
    echo "⚠ Missing prediction file: ${PRED_FILE}, skip..."
    continue
  fi

  echo ""
  echo "=========================================="
  echo "Evaluating ${SUBTASK}..."
  echo "=========================================="
  
  # Create a temporary Python script to evaluate this subtask
  # Use output directory for temp file to avoid permission issues
  eval_script="${OUT_DIR}/.eval_${SUBTASK}_$$.py"
  cat > "${eval_script}" << PYTHON_SCRIPT
import sys
import json
import os

# Add benchmark directory to path
sys.path.insert(0, "${BENCH_DIR}")

from eval.eval_molopt import eval_molopt_from_list, tranform_str_to_json

# Load predictions
with open("${PRED_FILE}", "r") as f:
    pred_results = json.load(f)

# Extract src_smiles and pred_smiles
src_smiles_list = []
pred_smiles_list = []
invalid_count = 0

for pred in pred_results:
    src_smiles = pred.get("src_smiles", "")
    if not src_smiles:
        invalid_count += 1
        continue
    
    json_results = pred.get("json_results", {})
    
    # Handle both dict and string formats
    final_target = None
    
    if isinstance(json_results, dict):
        final_target = json_results.get("Final Target Molecule") or json_results.get("Final_Target_Molecule")
    elif isinstance(json_results, str):
        # Try to parse string as JSON
        parsed_json = tranform_str_to_json(json_results)
        if parsed_json and isinstance(parsed_json, dict):
            final_target = parsed_json.get("Final Target Molecule") or parsed_json.get("Final_Target_Molecule")
        else:
            # Try direct regex extraction as fallback
            import re
            patterns = [
                r'"Final Target Molecule"\s*:\s*"([^"]+)"',
                r'"Final_Target_Molecule"\s*:\s*"([^"]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, json_results, re.IGNORECASE)
                if match:
                    final_target = match.group(1).strip()
                    break
    
    if final_target and final_target.strip():
        src_smiles_list.append(src_smiles)
        pred_smiles_list.append(final_target.strip())
    else:
        invalid_count += 1

print(f"Total predictions: {len(pred_results)}")
print(f"Valid predictions: {len(src_smiles_list)}")
print(f"Invalid predictions: {invalid_count}")

if len(src_smiles_list) == 0:
    print("⚠ No valid predictions found!")
    result_dict = {
        "error": "No valid predictions found",
        "total": len(pred_results),
        "valid": 0,
        "invalid": invalid_count
    }
else:
    # Evaluate using ChemCoTBench evaluator
    result_dict = eval_molopt_from_list(
        optimized_prop="${SUBTASK}",
        gt_list=src_smiles_list,
        pred_list=pred_smiles_list,
        total_number=len(pred_results)
    )
    result_dict["total_predictions"] = len(pred_results)
    result_dict["valid_predictions"] = len(src_smiles_list)
    result_dict["invalid_predictions"] = invalid_count

# Save results
with open("${OUT_FILE}", "w") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)

print(f"✓ Results saved to ${OUT_FILE}")
PYTHON_SCRIPT

  "${PYTHON_CMD}" "${eval_script}"
  # Clean up temporary script
  if [ -f "${eval_script}" ]; then
    rm -f "${eval_script}"
  fi
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in ${OUT_DIR}"
echo "=========================================="

# Print summary
echo ""
echo "Summary:"
for SUBTASK in logp drd jnk gsk qed solubility; do
  OUT_FILE="${OUT_DIR}/eval_${SUBTASK}.json"
  if [ -f "${OUT_FILE}" ]; then
    echo "  ${SUBTASK}: ${OUT_FILE}"
  fi
done

