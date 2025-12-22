import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_molecule_from_query(query: str) -> str:
    # Try to capture "Input Molecule: <smiles>" pattern
    m = re.search(r"Input Molecule:\s*([^,\n]+)", query)
    if m:
        return m.group(1).strip()
    # fallback: None
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_file", required=True, help="ChemCoTBench add/delete/sub.json")
    ap.add_argument("--output_file", required=True, help="Output json (list) matching ChemCoTBench logs format")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    data = load_json(args.input_file)
    results = []

    for ex in tqdm(data):
        query = ex.get("query") or ex.get("instruction") or ex.get("Instruction") or ""
        # prompt: use query原文，附加一句提醒输出 JSON
        prompt = query.strip()
        if not prompt.endswith("}"):
            prompt += "\nYour response must be a JSON: {\"output\": \"Modified Molecule SMILES\"}."

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)

        # 封装 ChemCoTBench 日志格式
        meta_str = ex.get("meta")
        meta = {}
        if isinstance(meta_str, str):
            try:
                meta = json.loads(meta_str)
            except Exception:
                meta = {}

        molecule = extract_molecule_from_query(query)
        task = ex.get("subtask") or ex.get("task")
        record = {
            "Instruction": query,
            "molecule": molecule,
            "added_group": meta.get("added_group"),
            "removed_group": meta.get("removed_group") or meta.get("deleted_group"),
            "reference": meta.get("reference"),
            "task": task,
            "json_results": f"```json\n{{\n    \"output\": \"{text}\"\n}}\n```",
        }
        results.append(record)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
