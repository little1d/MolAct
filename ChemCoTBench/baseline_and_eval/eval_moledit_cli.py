#!/usr/bin/env python3
"""
Command-line interface for evaluating molecule editing predictions.
Usage:
    python eval_moledit_cli.py --task mol_edit --subtask add --pred pred_add.json --out eval_add.json
"""

import json
import argparse
import sys
import os

# Add parent directory to path to import eval_moledit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_moledit import eval_moledit_from_list, tranform_str_to_json


def evaluate_moledit_file(pred_file, task, out_file):
    """
    Evaluate molecule editing predictions from a JSON file.
    
    Args:
        pred_file: Path to prediction JSON file
        task: Task type ('add', 'delete', or 'sub')
        out_file: Path to output JSON file
    """
    # Load predictions
    with open(pred_file, 'r') as f:
        pred_results = json.load(f)
    
    invalid_number = 0
    pred_list, src_list = list(), list()
    group_a, group_b = list(), list()
    
    for pred in pred_results:
        # Extract predicted SMILES
        if isinstance(pred.get('json_results'), str):
            pred_json = tranform_str_to_json(str_input=pred['json_results'])
            if pred_json is None:
                invalid_number += 1
                continue
            else:
                if 'output' not in pred_json.keys():
                    invalid_number += 1
                    continue
                pred_list.append(pred_json['output'])
        else:
            if isinstance(pred.get('json_results'), dict) and 'output' in pred['json_results']:
                pred_list.append(pred['json_results']['output'])
            else:
                invalid_number += 1
                continue
        
        # Extract source molecule - try multiple possible field names
        # Also try to extract from Instruction field if molecule field is empty
        src_mol = pred.get('molecule', '') or pred.get('src_smiles', '') or pred.get('source', '')
        
        # If still empty, try to extract from Instruction field
        if not src_mol:
            instruction = pred.get('Instruction', '')
            if 'Input Molecule:' in instruction:
                # Extract SMILES after "Input Molecule: "
                parts = instruction.split('Input Molecule:')
                if len(parts) > 1:
                    # Get the part after "Input Molecule:" and before the next comma or newline
                    mol_part = parts[1].split(',')[0].strip()
                    if mol_part:
                        src_mol = mol_part
        
        if not src_mol:
            invalid_number += 1
            if pred_list:  # Only pop if we added something
                pred_list.pop()  # Remove the last added pred
            continue
        src_list.append(src_mol)
        
        # Extract groups based on task
        if task == 'add':
            group_a.append(pred.get('added_group', ''))
        elif task == 'delete':
            group_a.append(pred.get('removed_group', ''))
        elif task == 'sub':
            group_a.append(pred.get('added_group', ''))
            group_b.append(pred.get('removed_group', ''))
    
    # Validate lengths
    assert len(src_list) == len(pred_list), f"Mismatch: {len(src_list)} src vs {len(pred_list)} pred"
    assert len(src_list) == len(group_a), f"Mismatch: {len(src_list)} src vs {len(group_a)} groups"
    if task == 'sub':
        assert len(group_a) == len(group_b), f"Mismatch: {len(group_a)} add vs {len(group_b)} remove"
    
    # Calculate total number (including invalid)
    total_number = len(pred_results)
    
    # Evaluate
    result_dict = eval_moledit_from_list(
        src_list=src_list,
        pred_list=pred_list,
        group_a=group_a,
        group_b=group_b if task == 'sub' else [],
        task=task,
        total_number=total_number
    )
    
    # Round results to 4 decimal places for readability
    rounded_result = {
        k: round(v, 4) if isinstance(v, (int, float)) else v
        for k, v in result_dict.items()
    }
    
    # Save results
    with open(out_file, 'w') as f:
        json.dump(rounded_result, f, indent=4)
    
    print(f"Evaluation completed for {task}")
    print(f"Results: {rounded_result}")
    print(f"Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate molecule editing predictions')
    parser.add_argument('--task', type=str, required=True, help='Task type (mol_edit)')
    parser.add_argument('--subtask', type=str, required=True, help='Subtask type (add, delete, or sub)')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction JSON file')
    parser.add_argument('--out', type=str, required=True, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    if args.task != 'mol_edit':
        print(f"Error: Only 'mol_edit' task is supported, got '{args.task}'")
        sys.exit(1)
    
    if args.subtask not in ['add', 'delete', 'sub']:
        print(f"Error: Subtask must be 'add', 'delete', or 'sub', got '{args.subtask}'")
        sys.exit(1)
    
    if not os.path.exists(args.pred):
        print(f"Error: Prediction file not found: {args.pred}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    
    evaluate_moledit_file(args.pred, args.subtask, args.out)


if __name__ == '__main__':
    main()

