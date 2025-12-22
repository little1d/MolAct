"""
Inference script for ChemCoTBench mol_edit using AgentFly ReactAgent with tools.

Requirements:
- AgentFly installed in the repo (imports from agentfly.agents.react.react_agent, TOOL_REGISTRY).
- A merged HF checkpoint directory passed via --model_path.
- ChemCoTBench mol_edit json (add/delete/sub) passed via --input_file.

This script builds a ReactAgent with the chemistry tools registered in TOOL_REGISTRY
and runs a single turn per query. The final assistant message is expected to
contain a JSON like {"output": "<SMILES>"}; we extract that and save in the
ChemCoTBench log format.
"""

import os
import sys
from pathlib import Path

# Note: verl is not required for inference with transformers backend
# verl is only needed if using async_verl backend (training setup)

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


def extract_output(text: str) -> str:
    """Extract the last {"output": "..."} or code block JSON from text."""
    if not text:
        return ""
    
    # code block json
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if blocks:
        for blk in reversed(blocks):
            try:
                obj = json.loads(blk)
                if "output" in obj:
                    return obj["output"].strip()
            except Exception:
                continue
    
    # plain json (more flexible pattern)
    matches = re.findall(r"\{[^{}]*\"output\"[^{}]*\}", text, flags=re.DOTALL)
    if matches:
        for m in reversed(matches):
            try:
                obj = json.loads(m)
                if "output" in obj:
                    return obj["output"].strip()
            except Exception:
                continue
    
    # Try to find Answer: prefix (ReAct format)
    answer_match = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Try to extract JSON from answer
        json_match = re.search(r"\{[^{}]*\"output\"[^{}]*\}", answer_text, flags=re.DOTALL)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if "output" in obj:
                    return obj["output"].strip()
            except Exception:
                pass
        return answer_text
    
    # Fallback: try simple pattern
    m2 = re.search(r"output\"?\s*:\s*\"([^\"]+)\"", text)
    if m2:
        return m2.group(1).strip()
    
    return text.strip()


def build_messages(query: str) -> List[Dict[str, Any]]:
    # Only user message; system prompt comes from template / agent defaults.
    user = {"role": "user", "content": query}
    return [user]


def make_agent(model_path: str, temperature: float, top_p: float, backend: str = "transformers"):
    """
    Create a ReactAgent with specified backend.
    
    Args:
        model_path: Path to the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        backend: Backend to use. Options:
            - "transformers": HuggingFace Transformers (no multiprocessing, slower but more compatible)
            - "async_verl": verl backend (used in training, requires verl environment)
    """
    agent = ReactAgent(
        model_name_or_path=model_path,
        template="qwen2.5-no-system-tool",
        tools=list(TOOL_REGISTRY.values()),
        backend=backend,
        generation_config={
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    return agent


async def process_single_example(
    agent: ReactAgent,
    ex: Dict[str, Any],
    args: argparse.Namespace,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Process a single example and return the result record."""
    query = ex.get("query") or ex.get("instruction") or ex.get("Instruction") or ""
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
    
    # Extract output SMILES from the assistant message
    raw_text = assistant_msg
    output_smiles = extract_output(raw_text)
    
    # Parse meta information
    meta_str = ex.get("meta")
    meta = {}
    if isinstance(meta_str, str):
        try:
            meta = json.loads(meta_str)
        except Exception:
            meta = {}
    
    record = {
        "Instruction": query,
        "molecule": ex.get("src") or ex.get("source") or "",
        "added_group": meta.get("added_group"),
        "removed_group": meta.get("removed_group") or meta.get("deleted_group"),
        "reference": meta.get("reference"),
        "task": ex.get("subtask") or ex.get("task"),
        "json_results": f"```json\n{{\n    \"output\": \"{output_smiles}\"\n}}\n```",
        "assistant_raw": raw_text,
        "trajectory": traj,
    }
    return record


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_file", required=True, help="ChemCoTBench add/delete/sub json")
    ap.add_argument("--output_file", required=True, help="Output json list matching ChemCoTBench logs format")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_turns", type=int, default=16)
    ap.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "async_verl"],
        help="Backend to use. 'transformers' is the default for inference. 'async_verl' requires verl environment (training setup).",
    )
    args = ap.parse_args()

    data = load_data(args.input_file)
    print(f"Loaded {len(data)} examples from {args.input_file}")
    
    # Check if verl is available when using async_verl backend
    verl_available = False
    if args.backend == "async_verl":
        try:
            import verl
            from verl.protocol import DataProto
            verl_available = True
            print("✓ verl module is available.")
        except ImportError as e:
            print(f"⚠ WARNING: verl module is not available: {e}")
            print("  async_verl backend requires verl to be installed and accessible.")
            print("  Switching to 'transformers' backend for inference.")
            print("  Note: transformers backend supports tools but may be slower.")
            args.backend = "transformers"
            verl_available = False
    
    agent = make_agent(
        args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        backend=args.backend,
    )
    print(f"Initialized agent with model: {args.model_path}, backend: {args.backend}")
    
    # For async_verl backend, check if engine is set
    if args.backend == "async_verl" and agent.llm_engine.llm_engine is None:
        print("\n" + "="*60)
        print("⚠ ERROR: async_verl backend requires a verl engine to be set.")
        print("="*60)
        print("The async_verl backend needs a verl rollout engine which is typically")
        print("created during training with Ray worker groups.")
        print("\nTo use async_verl backend, you need to:")
        print("  1. Set up verl training environment with Ray")
        print("  2. Create a verl rollout worker/engine")
        print("  3. Set it via: agent.set_llm_engine(verl_engine, tokenizer, processor)")
        print("\nFor simple inference, transformers backend also supports tools.")
        print("Switching to 'transformers' backend...")
        print("="*60 + "\n")
        # Recreate agent with transformers backend
        agent = make_agent(
            args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            backend="transformers",
        )
        print(f"✓ Re-initialized agent with backend: transformers")
        print("  Note: Tools are supported, but inference may be slower than verl backend.\n")

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
