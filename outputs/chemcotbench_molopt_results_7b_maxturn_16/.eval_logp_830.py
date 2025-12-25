import sys
import json
import os

# Add benchmark directory to path
sys.path.insert(0, "/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/ChemCoTBench/baseline_and_eval")

from eval.eval_molopt import eval_molopt_from_list, tranform_str_to_json

# Load predictions
with open("/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/mol_opt_eval_bench_7b_maxturn_16//pred_logp.json", "r") as f:
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
        optimized_prop="logp",
        gt_list=src_smiles_list,
        pred_list=pred_smiles_list,
        total_number=len(pred_results)
    )
    result_dict["total_predictions"] = len(pred_results)
    result_dict["valid_predictions"] = len(src_smiles_list)
    result_dict["invalid_predictions"] = invalid_count

# Save results
with open("/mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/chemcotbench_molopt_results_7b_maxturn_16/eval_logp.json", "w") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)

print(f"✓ Results saved to /mnt/shared-storage-user/yangzhuo/main/projects/MolAct/outputs/chemcotbench_molopt_results_7b_maxturn_16/eval_logp.json")
