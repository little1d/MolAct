"""
Inference script for ChemCoTBench mol_opt using AgentFly ReactAgent with tools.

Requirements:
- AgentFly installed in the repo (imports from agentfly.agents.react.react_agent, TOOL_REGISTRY).
- A merged HF checkpoint directory passed via --model_path.
- ChemCoTBench mol_opt json (logp/drd/jnk/gsk/qed/solubility) passed via --input_file.

This script builds a ReactAgent with the chemistry tools registered in TOOL_REGISTRY
and runs inference per query. The final assistant message is expected to
contain a JSON like {"Final Target Molecule": "<SMILES>"}; we extract that and save in thes
ChemCoTBench log format.
"""

import os
import sys
from pathlib import Path

# No need to add verl directory to sys.path since we're only using TransformersBackend

# Now safe to import other modules
import argparse
import json
import re
import asyncio
from typing import Any, Dict, List, Optional

from agentfly.agents.react.react_agent import ReactAgent
from agentfly.tools import TOOL_REGISTRY


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_final_target_molecule(text: str) -> Optional[str]:
    """Extract 'Final Target Molecule' from text, handling various formats."""
    if not text:
        return None
    
    # Try to extract from code block json first
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for blk in reversed(blocks):
        try:
            obj = json.loads(blk)
            if "Final Target Molecule" in obj:
                return obj["Final Target Molecule"].strip()
            if "Final_Target_Molecule" in obj:
                return obj["Final_Target_Molecule"].strip()
        except Exception:
            continue
    
    # Try to find JSON object with Final Target Molecule
    # Match both "Final Target Molecule" and "Final_Target_Molecule"
    patterns = [
        r'"Final Target Molecule"\s*:\s*"([^"]+)"',
        r'"Final_Target_Molecule"\s*:\s*"([^"]+)"',
        r'"Final Target Molecule"\s*:\s*"([^"]+)"',
        r'"Final_Target_Molecule"\s*:\s*"([^"]+)"',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return the last match
    
    # Try to find in plain JSON (more flexible)
    json_matches = re.findall(r"\{[^{}]*\"Final[^{}]*Target[^{}]*Molecule\"[^{}]*\}", text, flags=re.DOTALL | re.IGNORECASE)
    for match in reversed(json_matches):
        try:
            obj = json.loads(match)
            for key in obj.keys():
                if "final" in key.lower() and "target" in key.lower() and "molecule" in key.lower():
                    return str(obj[key]).strip()
        except Exception:
            continue
    
    return None


def extract_src_smiles(query: str) -> Optional[str]:
    """Extract source SMILES from query text."""
    # Look for "Source Molecule: <SMILES>"
    pattern = r"Source Molecule:\s*([A-Za-z0-9@\[\]()=+\-\\/]+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def build_messages(query: str) -> List[Dict[str, Any]]:
    # Only user message; system prompt comes from template / agent defaults.
    user = {"role": "user", "content": query}
    return [user]


def make_agent(model_path: str, temperature: float, top_p: float):
    """
    Create a ReactAgent with TransformersBackend.
    
    Args:
        model_path: Path to the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    """
    agent = ReactAgent(
        model_name_or_path=model_path,
        template="qwen2.5-no-system-tool",
        tools=list(TOOL_REGISTRY.values()),
        backend="transformers",
        generation_config={
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        },
        tokenizer_kwargs={"fix_mistral_regex": True},
    )
    return agent


async def process_single_example(
    agent: ReactAgent,
    ex: Dict[str, Any],
    args: argparse.Namespace,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Process a single example and return the result record."""
    query = ex.get("query") or ex.get("question") or ""
    if not query:
        print(f"Warning: Empty query for entry {idx}, skipping...")
        return None
    
    messages = build_messages(query)
    
    try:
        # agent.run is async and doesn't return trajectory directly
        await agent.run(
            messages=messages,
            max_turns=args.max_turns,
            num_chains=1,
            generation_config={"max_new_tokens": args.max_new_tokens},
        )
        # Get messages from the agent after run completes
        traj = agent.get_messages()  # Returns [{"messages": [...], ...}]
    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        traj = []
    
    # Extract last assistant message content
    assistant_msg = ""
    if isinstance(traj, list) and len(traj) > 0:
        # traj is a list of chain results, each containing messages
        # Get the first chain's messages (since num_chains=1)
        chain_data = traj[0] if isinstance(traj[0], dict) else {}
        chain_messages = chain_data.get("messages", [])
        
        if not isinstance(chain_messages, list):
            chain_messages = [chain_messages] if chain_messages else []
        
        # Find last assistant message
        for msg in reversed(chain_messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list) and content:
                    # Handle structured content (e.g., [{"type": "text", "text": "..."}])
                    for item in content:
                        if isinstance(item, dict):
                            assistant_msg = item.get("text", "") or assistant_msg
                        elif isinstance(item, str):
                            assistant_msg = item or assistant_msg
                elif isinstance(content, str):
                    assistant_msg = content
                if assistant_msg:
                    break
    
    # Extract Final Target Molecule from the assistant message
    raw_text = assistant_msg
    final_target_molecule = extract_final_target_molecule(raw_text)
    
    # Extract source SMILES from query or use provided src_smiles
    src_smiles = ex.get("src_smiles") or extract_src_smiles(query) or ""
    
    # Get subtask/prop
    subtask = ex.get("subtask") or ex.get("prop") or ""
    
    # Format json_results similar to ChemCoTBench logs format
    if final_target_molecule:
        json_results = {
            "Final Target Molecule": final_target_molecule
        }
    else:
        # If extraction failed, save raw text (might be a string with JSON)
        json_results = raw_text if raw_text else ""
    
    record = {
        "src_smiles": src_smiles,
        "prop": subtask,
        "json_results": json_results,
        "trajectory": traj,  # Save full trajectory like mol_edit
    }
    return record


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_file", required=True, help="ChemCoTBench mol_opt json (logp/drd/jnk/gsk/qed/solubility)")
    ap.add_argument("--output_file", required=True, help="Output json list matching ChemCoTBench logs format")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_turns", type=int, default=16)
    args = ap.parse_args()

    data = load_data(args.input_file)
    print(f"Loaded {len(data)} examples from {args.input_file}")
    
    # Create agent with TransformersBackend
    agent = make_agent(
        args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Initialized agent with model: {args.model_path}, backend: transformers")

    results = []
    for idx, ex in enumerate(data):
        if (idx + 1) % 10 == 0:
            print(f"Processing {idx + 1}/{len(data)}...")
        
        record = await process_single_example(agent, ex, args, idx)
        if record:
            results.append(record)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(results)} results to {out_path}")


def main():
    """Entry point that handles async execution."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

