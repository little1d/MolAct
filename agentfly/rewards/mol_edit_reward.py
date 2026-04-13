from agentfly.rewards import reward
from agentfly.tools import TOOL_REGISTRY
from rdkit import Chem
import re
import json
import sys
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# Debug file for reward function
REWARD_DEBUG_FILE = os.environ.get("REWARD_DEBUG_FILE", "/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/logs/reward_debug.log")

LOG_STRIDE = 200  # log roughly every LOG_STRIDE calls
_log_counter = 0
_chain_collector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_expected_chains_per_group = 4  # default num_chains
# Support environment variable for log file path (useful for running multiple experiments)
LOG_FILE = Path(os.environ.get("MOL_EDIT_TRAJ_FILE", "logs/mol_edit_traj.jsonl"))


MIN_SMILES_LEN = 2
# ChemCoTBench functional group definitions
GROUP_SET = {
    "benzene",
    "benzene_ring",
    "hydroxyl",
    "anhydride",
    "aldehyde",
    "ketone",
    "carboxyl",
    "ester",
    "amide",
    "amine",
    "nitro",
    "halo",
    "thioether",
    "nitrile",
    "thiol",
    "sulfide",
    "disulfide",
    "sulfoxide",
    "sulfone",
    "borane",
}

GROUP_TO_SMARTS = {
    "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "hydroxyl": "[OX2H]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "halo": "[F,Cl,Br,I]",
    "thiol": "[#16X2H]",
    "thioether": "[SX2][CX4]",
    "disulfide": "[#16X2H0][#16X2H0]",
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "sulfide": "[#16X2H0]",
    "nitrile": "[NX1]#[CX2]",
    "borane": "[BX3]",
}


def _get_last_text_by_role(role: str, trajectory: Any) -> str:
    """
    Fetch the latest message content for a given role from the trajectory.
    Only pulls the first text field if content is a list of parts.
    """
    if trajectory is None:
        return ""
    msgs = trajectory.get("messages", []) if isinstance(trajectory, dict) else trajectory
    if not isinstance(msgs, list):
        return ""
    for msg in reversed(msgs):
        if msg.get("role") == role:
            content = msg.get("content", "")
            if isinstance(content, list) and content:
                return content[0].get("text", "") or ""
            if isinstance(content, str):
                return content
            return ""
    return ""


def _get_last_tool_observation(trajectory: Any) -> str:
    """
    Get the latest tool observation text, if any.
    """
    if trajectory is None:
        return ""
    msgs = trajectory.get("messages", []) if isinstance(trajectory, dict) else trajectory
    if not isinstance(msgs, list):
        return ""
    for msg in reversed(msgs):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    return first.get("text", "") or first.get("observation", "") or ""
                if isinstance(first, str):
                    return first
            if isinstance(content, dict):
                return content.get("text", "") or content.get("observation", "") or ""
            if isinstance(content, str):
                return content
    return ""


def extract_last_valid_smiles(text: str, trajectory: Any = None) -> Optional[str]:
    """
    Extract a valid SMILES with priority:
      1) Tool observations (especially chem_edit_smiles results) - HIGHEST PRIORITY
      2) <smiles>...</smiles> tags in latest assistant message
      3) Raw prediction fallback
    """
    candidates: list[tuple[str, int]] = []  # (smiles, priority)

    def is_valid_smiles_string(s: str) -> bool:
        """Check if string looks like a valid SMILES (no spaces, no words)."""
        if not s or len(s) < MIN_SMILES_LEN:
            return False
        # SMILES should not contain spaces or common English words
        if ' ' in s or any(word in s.lower() for word in ['thought', 'action', 'answer', 'input', 'error', 'success']):
            return False
        # Must start with a valid SMILES character
        if not s[0].isalpha() and s[0] not in '[(':
            return False
        return True

    def add_candidates_from_source(source: str, priority: int, skip_action_lines: bool = True):
        if not source:
            return
        # 过滤掉明显的工具输入行
        if skip_action_lines:
            filtered = "\n".join([ln for ln in source.splitlines() 
                                  if "Input:" not in ln and "Action:" not in ln])
        else:
            filtered = source
        
        # 优先提取 <smiles> 标签
        tags = re.findall(r"<smiles>(.*?)</smiles>", filtered, flags=re.DOTALL | re.IGNORECASE)
        for t in tags:
            cand = t.strip(" \n\t.,;:!\"'`")
            if is_valid_smiles_string(cand):
                candidates.append((cand, priority + 1))
        
        # 提取 "New SMILES: XXX" 格式
        new_smiles_match = re.search(r"New SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)\\\/%=#$\.\,]+)", filtered)
        if new_smiles_match:
            cand = new_smiles_match.group(1).strip(".,;:!\"'` ")
            if is_valid_smiles_string(cand):
                candidates.append((cand, priority + 2))
        
        # 正则提取其他候选（较低优先级）
        pattern = r"[A-Za-z0-9@+\-\[\]\(\)\\\/%=#$]{5,}"
        for match in re.finditer(pattern, filtered):
            cand = match.group(0).strip(".,;:!\"'` ")
            if is_valid_smiles_string(cand):
                candidates.append((cand, priority))

    # 1) 先从 assistant 消息中提取（包括 tool call Input 中的 SMILES）
    assistant_text = _get_last_text_by_role("assistant", trajectory)
    if assistant_text:
        # 先从 assistant 文本中提取（跳过 Action/Input 行，避免提取工具输入）
        add_candidates_from_source(assistant_text, 5, skip_action_lines=True)
        # 然后也从 assistant 文本中提取 tool call Input 中的 SMILES（特别是 chem_add_group 等工具调用后的新 SMILES）
        # 这些工具调用后，agent 通常会使用新的 SMILES 调用其他工具（如 chem_calc_logp）
        tool_input_pattern = r'Input:\s*\{[^}]*"smiles":\s*"([A-Za-z0-9@+\-\[\]\(\)\\\/%=#$\.\,]+)"'
        tool_input_matches = re.finditer(tool_input_pattern, assistant_text, re.IGNORECASE)
        for match in tool_input_matches:
            cand = match.group(1).strip(".,;:!\"'` ")
            if is_valid_smiles_string(cand):
                # 如果这个 SMILES 出现在 chem_add_group/remove_group/replace_group 之后，给予更高优先级
                pos = match.start()
                before_text = assistant_text[max(0, pos-500):pos].lower()
                if any(tool in before_text for tool in ["chem_add_group", "chem_remove_group", "chem_replace_group", "successfully added", "successfully removed", "successfully replaced"]):
                    candidates.append((cand, 6))  # 比普通 assistant 文本稍高优先级
                else:
                    candidates.append((cand, 4))  # 比普通 assistant 文本稍低优先级
    
    # 2) 然后从工具返回（tool observations）中提取（最高优先级，因为这是工具实际返回的结果）
    if trajectory:
        msgs = trajectory.get("messages", []) if isinstance(trajectory, dict) else trajectory
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if msg.get("role") == "tool":
                    tool_name = msg.get("tool_name") or msg.get("name", "")
                    content = msg.get("content", "")
                    
                    # 提取工具返回的文本
                    tool_text = ""
                    if isinstance(content, list) and content:
                        first = content[0]
                        if isinstance(first, dict):
                            tool_text = first.get("text", "") or first.get("observation", "") or ""
                        elif isinstance(first, str):
                            tool_text = first
                    elif isinstance(content, dict):
                        tool_text = content.get("text", "") or content.get("observation", "") or ""
                    elif isinstance(content, str):
                        tool_text = content
                    
                    # chem_edit_smiles (legacy) or new tools success result has highest priority
                    if tool_name in ["chem_edit_smiles", "chem_add_group", "chem_remove_group", "chem_replace_group"] and "Success" in tool_text:
                        add_candidates_from_source(tool_text, 10, skip_action_lines=False)
                        break
                    elif tool_text and "Error" not in tool_text:
                        add_candidates_from_source(tool_text, 8, skip_action_lines=False)
    
    # 3) raw prediction fallback
    add_candidates_from_source(text, 3)

    # 按优先级排序，选择最佳候选
    if candidates:
        # 按优先级降序，长度降序排序
        candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        for cand, _ in candidates:
            if len(cand) < MIN_SMILES_LEN:
                continue
            # 严格验证 SMILES
            try:
                mol = Chem.MolFromSmiles(cand)
                if mol is not None and mol.GetNumAtoms() > 0:
                    return cand
            except Exception:
                continue
    
    return None


# ---------- ChemCoTBench-style checks ----------
def _mol_prop(mol: str, prop: str) -> Optional[int]:
    try:
        m = Chem.MolFromSmiles(mol)
    except Exception:
        return None
    if m is None:
        return None
    if prop.startswith("num_"):
        smarts = GROUP_TO_SMARTS.get(prop.replace("num_", ""), None)
        if smarts is None:
            return None
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return None
        matches = m.GetSubstructMatches(patt)
        if prop == "num_sulfide":
            # subtract disulfide occurrences
            exc = Chem.MolFromSmarts("[#16X2H0][#16X2H0]")
            if exc:
                exc_matches = m.GetSubstructMatches(exc)
                return len(matches) - len(exc_matches)
        elif prop == "num_hydroxyl":
            # subtract carboxyl occurrences (since carboxyl contains hydroxyl)
            carboxyl_smarts = GROUP_TO_SMARTS.get("carboxyl")
            if carboxyl_smarts:
                carboxyl_patt = Chem.MolFromSmarts(carboxyl_smarts)
                if carboxyl_patt:
                    carboxyl_matches = m.GetSubstructMatches(carboxyl_patt)
                    return max(0, len(matches) - len(carboxyl_matches))
        return len(matches)
    return None


def _check_add(src: str, tgt: str, group: str) -> bool:
    if group not in GROUP_SET:
        return False
    a = _mol_prop(src, f"num_{group}")
    b = _mol_prop(tgt, f"num_{group}")
    if a is None or b is None:
        return False
    return b == a + 1


def _check_del(src: str, tgt: str, group: str) -> bool:
    if group not in GROUP_SET:
        return False
    a = _mol_prop(src, f"num_{group}")
    b = _mol_prop(tgt, f"num_{group}")
    if a is None or b is None:
        return False
    return b == a - 1


def _check_sub(src: str, tgt: str, remove_group: str, add_group: str) -> bool:
    if remove_group not in GROUP_SET or add_group not in GROUP_SET:
        return False
    a_rm = _mol_prop(src, f"num_{remove_group}")
    b_rm = _mol_prop(tgt, f"num_{remove_group}")
    a_add = _mol_prop(src, f"num_{add_group}")
    b_add = _mol_prop(tgt, f"num_{add_group}")
    if None in (a_rm, b_rm, a_add, b_add):
        return False
    return b_rm == a_rm - 1 and b_add == a_add + 1


def _extract_group_id(trajectory: Any, **kwargs) -> Optional[str]:
    if isinstance(trajectory, dict):
        meta = trajectory.get("meta") or {}
        if "group_id" in meta:
            return meta["group_id"]
        msgs = trajectory.get("messages", [])
        if msgs and isinstance(msgs[0], dict):
            meta0 = msgs[0].get("meta")
            if meta0 and "group_id" in meta0:
                return meta0["group_id"]
    return kwargs.get("group_id") or "default_group"


def _log_chain_group(group_id: str, chains: List[Dict[str, Any]], ref_smiles: Optional[str] = None):
    _dump_group_to_file(group_id, chains, ref_smiles)


def _dump_group_to_file(group_id: str, chains: List[Dict[str, Any]], ref_smiles: Optional[str]):
    # Check if trajectory logging is disabled
    if os.environ.get("DISABLE_TRAJ_LOG", "0") == "1":
        return
    
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "group_id": group_id,
            "ref_smiles": ref_smiles,
            "chains": [],
        }
        for info in chains:
            payload["chains"].append(
                {
                    "reward": info.get("reward"),
                    "valid": info.get("valid"),
                    "sim": info.get("sim"),
                    "extracted": info.get("extracted"),
                    "raw_pred": info.get("raw_pred"),
                    "trajectory": info.get("trajectory"),
                    "debug_info": info.get("debug_info"),
                }
            )
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[mol_edit_simple] failed to write log file: {e}")


@reward(name="mol_edit_simple")
async def mol_edit_simple(prediction: str, trajectory, ref_smiles: str = None, num_chains: int = None, **kwargs):
    """
    Reward for molecule editing:
    - invalid SMILES -> -1
    - valid -> base 0.1 + similarity component + tool usage bonus
    """
    # If ref_smiles is not provided as a parameter, try to get it from kwargs
    if ref_smiles is None:
        ref_smiles = kwargs.get("ref_smiles")
    
    global _log_counter, _chain_collector, _expected_chains_per_group
    if num_chains:
        _expected_chains_per_group = num_chains
    _log_counter += 1
    do_log = (_log_counter % LOG_STRIDE == 0)

    # Debug: Print kwargs at function entry (before any processing)
    # Print for first 10 calls, then every LOG_STRIDE calls
    do_debug_entry = (_log_counter % LOG_STRIDE == 0) or (_log_counter < 10)
    do_debug_print = do_debug_entry  # Alias for consistency
    if do_debug_entry:
        debug_entry_msg = f"[mol_edit_reward Debug] Entry for Call #{_log_counter}:\n"
        debug_entry_msg += f"  kwargs keys at entry: {list(kwargs.keys())}\n"
        debug_entry_msg += f"  kwargs values at entry: {kwargs}\n"
        # Also print kwargs items individually for better visibility
        if kwargs:
            debug_entry_msg += f"  kwargs items:\n"
            for key, value in kwargs.items():
                debug_entry_msg += f"    {key}: {value}\n"
        print(debug_entry_msg, flush=True)
        try:
            with open(REWARD_DEBUG_FILE, "a") as f:
                f.write(debug_entry_msg + "\n")
        except Exception:
            pass

    # 初始化默认字段，确保 batch 对齐
    result = {
        "reward": -1.0,
        "valid": 0,
        "sim": -1.0,
        "mw": 0.0,
        "logp": 0.0,
        "qed": 0.0,
        "sa": 0.0,
        "correct": 0.0,
        "tool_bonus": 0.0,
        "sim_score": 0.0,
        "debug_info": "",
    }

    try:
        # Try to extract a valid SMILES if the raw prediction contains extra text
        extracted = extract_last_valid_smiles(prediction, trajectory=trajectory)
        
        # Debug: Print extraction result (only for LOG_STRIDE calls)
        if _log_counter % LOG_STRIDE == 0:
            debug_extract_msg = f"[mol_edit_reward Debug] Extraction result for Call #{_log_counter}:\n"
            debug_extract_msg += f"  prediction (first 200 chars): {prediction[:200] if prediction else 'None'}...\n"
            debug_extract_msg += f"  extracted: {extracted}\n"
            debug_extract_msg += f"  trajectory type: {type(trajectory)}\n"
            print(debug_extract_msg, flush=True)
            try:
                with open(REWARD_DEBUG_FILE, "a") as f:
                    f.write(debug_extract_msg + "\n")
            except Exception:
                pass

        def _first(x):
            if isinstance(x, (list, tuple)) and x:
                return x[0]
            return x

        # Debug: Print kwargs before processing (only for LOG_STRIDE calls)
        if _log_counter % LOG_STRIDE == 0:
            debug_msg = f"[mol_edit_reward Debug] Call #{_log_counter}:\n"
            debug_msg += f"  kwargs keys: {list(kwargs.keys())}\n"
            debug_msg += f"  kwargs values: {kwargs}\n"
            print(debug_msg, flush=True)
            logger.warning(debug_msg)
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Also write to debug file
            try:
                with open(REWARD_DEBUG_FILE, "a") as f:
                    f.write(debug_msg + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Failed to write debug file: {e}")
        
        # Note: Most fields are passed directly in kwargs from naive.py
        # However, ref_smiles might still be in extra_info or reward_model for backward compatibility
        # Extract ref_smiles from extra_info or reward_model if not in kwargs
        if "ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None:
            # Try from extra_info
            extra_info = kwargs.get("extra_info", {})
            if isinstance(extra_info, (list, tuple)) and extra_info and isinstance(extra_info[0], dict):
                extra_info = extra_info[0]
            if isinstance(extra_info, dict) and "ref_smiles" in extra_info:
                kwargs["ref_smiles"] = extra_info["ref_smiles"]
            # Try from reward_model.ground_truth
            if ("ref_smiles" not in kwargs or kwargs.get("ref_smiles") is None):
                reward_model = kwargs.get("reward_model", {})
                if isinstance(reward_model, dict) and "ground_truth" in reward_model:
                    kwargs["ref_smiles"] = reward_model["ground_truth"]

        # 顶层字段可能仍为 list/tuple，取第一个
        for key in ["task", "subtask", "src_smiles", "add_group", "remove_group", "ref_smiles"]:
            if key in kwargs:
                kwargs[key] = _first(kwargs[key])

        # Debug: Print kwargs after _first processing
        if do_debug_print:
            debug_msg3 = f"[mol_edit_reward Debug] After _first processing:\n"
            debug_msg3 += f"  task: {kwargs.get('task')}, subtask: {kwargs.get('subtask')}\n"
            debug_msg3 += f"  src_smiles: {kwargs.get('src_smiles')}\n"
            debug_msg3 += f"  add_group: {kwargs.get('add_group')}, remove_group: {kwargs.get('remove_group')}\n"
            debug_msg3 += f"  ref_smiles: {kwargs.get('ref_smiles')}\n"
            debug_msg3 += f"  extracted: {extracted}\n"
            print(debug_msg3, flush=True)
            try:
                with open(REWARD_DEBUG_FILE, "a") as f:
                    f.write(debug_msg3 + "\n")
            except Exception:
                pass

        if extracted:
            # 如果提取结果与输入分子相同且提供了 src_smiles，则判定为无效编辑
            src_smiles = kwargs.get("src_smiles") or kwargs.get("source_smiles")
            if src_smiles and extracted.strip() == src_smiles.strip():
                return result

            validate = await TOOL_REGISTRY["chem_mol_validate"](smiles=extracted)
            valid_flag = validate.get("valid", None)
            if valid_flag is None:
                valid_flag = validate.get("info", {}).get("valid", None)
            if valid_flag == 1:
                props = await TOOL_REGISTRY["chem_calc_properties"](smiles=extracted, ref_smiles=ref_smiles)
                info = props.get("info", props)
                sim = info.get("sim", None)
                sim_score = max(sim, 0.0) if sim is not None else 0.0

                # Use task directly (subtask is always identical to task)
                task = kwargs.get("task")
                # Normalize task value: handle "substitute" -> "sub"
                if task and isinstance(task, str):
                    task = task.lower().strip()
                    if task in ["substitute", "substitution"]:
                        task = "sub"
                
                remove_group = kwargs.get("remove_group") or kwargs.get("group_b")
                add_group = kwargs.get("add_group") or kwargs.get("group_a")
                correct = 0.0
                
                # Track missing fields for debugging
                missing_fields = []
                if not task:
                    missing_fields.append("MISSING_TASK")
                if not src_smiles:
                    missing_fields.append("MISSING_SRC_SMILES")
                if task == "add" and not add_group:
                    missing_fields.append("MISSING_ADD_GROUP")
                if task == "delete" and not remove_group:
                    missing_fields.append("MISSING_REMOVE_GROUP")
                if task == "sub" and (not add_group or not remove_group):
                    missing_fields.append(f"MISSING_SUB_GROUPS(add={add_group}, rm={remove_group})")
                
                if task and src_smiles:
                    if task == "add" and add_group:
                        correct = 1.0 if _check_add(src_smiles, extracted, add_group) else 0.0
                    elif task == "delete" and remove_group:
                        correct = 1.0 if _check_del(src_smiles, extracted, remove_group) else 0.0
                    elif task == "sub" and add_group and remove_group:
                        correct = 1.0 if _check_sub(src_smiles, extracted, remove_group, add_group) else 0.0
                
                # Debug info for correct score
                missing_str = ", ".join(missing_fields) if missing_fields else ""
                debug_info = (
                    f"task={task}, subtask={kwargs.get('subtask')}, "
                    f"src_smiles={kwargs.get('src_smiles')}, ref_smiles={kwargs.get('ref_smiles')}, "
                    f"add={add_group}, rm={remove_group}, correct={correct}, "
                    f"kwargs_keys={list(kwargs.keys())}"
                )
                if missing_str:
                    debug_info += f", {missing_str}"
                
                # Debug: Print correct calculation details
                if do_debug_print:
                    debug_msg3 = f"[mol_edit_reward Debug] Correct calculation:\n"
                    debug_msg3 += f"  task: {task}, src_smiles: {src_smiles}\n"
                    debug_msg3 += f"  add_group: {add_group}, remove_group: {remove_group}\n"
                    debug_msg3 += f"  correct: {correct}, missing_fields: {missing_fields}\n"
                    debug_msg3 += f"  extracted: {extracted}\n"
                    print(debug_msg3, flush=True)
                    logger.warning(debug_msg3)
                    try:
                        with open(REWARD_DEBUG_FILE, "a") as f:
                            f.write(debug_msg3 + "\n")
                            f.flush()
                    except Exception:
                        pass

                used_tool = _check_tool_used(trajectory, prediction)
                tool_bonus = 0.05 if used_tool else 0.0  # 较小权重
                reward_val = 0.8 * correct + 0.15 * sim_score + tool_bonus
                reward_val = max(-1.0, min(reward_val, 1.0))
                result.update(
                    {
                        "reward": float(reward_val),
                        "valid": 1,
                        "sim": float(sim) if sim is not None else -1.0,
                        "mw": float(info.get("mw") or 0.0),
                        "logp": float(info.get("logp") or 0.0),
                        "qed": float(info.get("qed") or 0.0),
                        "sa": float(info.get("sa") or 0.0),
                        "correct": float(correct),
                        "tool_bonus": float(tool_bonus),
                        "sim_score": float(sim_score),
                        "debug_info": debug_info,
                    }
                )
    except Exception as e:
        # Fail gracefully; result stays as default for batch alignment
        result["debug_info"] = f"Exception: {str(e)}"
        pass

    # 如果没有命中任何路径，仍然输出调试信息方便排查
    if not result.get("debug_info"):
        result["debug_info"] = (
            f"extracted={extracted if 'extracted' in locals() else None}, "
            f"task={kwargs.get('task')}, subtask={kwargs.get('subtask')}, "
            f"src_smiles={kwargs.get('src_smiles')}, ref_smiles={kwargs.get('ref_smiles')}, "
            f"add={kwargs.get('add_group')}, rm={kwargs.get('remove_group')}, "
            f"kwargs_keys={list(kwargs.keys())}"
        )

    # 用于日志的提取结果需要在 try 块后定义
    # 记录日志用的抽取结果（如果有效则再抽取一次，确保不为 None）
    extracted_for_log = ""
    if "extracted" in locals():
        extracted_for_log = extracted or ""
    if result["valid"] == 1:
        extracted_for_log = extract_last_valid_smiles(prediction, trajectory=trajectory) or extracted_for_log

    group_id = _extract_group_id(trajectory, **kwargs) or f"unknown_{_log_counter}"
    _chain_collector[group_id].append(
        {
            "reward": result["reward"],
            "valid": result["valid"],
            "sim": result.get("sim", -1.0),
            "extracted": extracted_for_log,
            "raw_pred": prediction,
            "trajectory": trajectory,
            "debug_info": result.get("debug_info", ""),
        }
    )
    # log once per group when group is complete; save all chains for the group
    reached_group = len(_chain_collector[group_id]) >= _expected_chains_per_group
    if reached_group:
        # Save all chains for this group (not just one sample)
        _log_chain_group(group_id, _chain_collector[group_id], ref_smiles)
        del _chain_collector[group_id]
    # Also periodically save incomplete groups to ensure data is not lost
    elif do_log and len(_chain_collector[group_id]) > 0:
        # Save current progress for this group even if not complete
        _log_chain_group(group_id, _chain_collector[group_id], ref_smiles)
    return result


def _check_tool_used(trajectory: Any, raw_pred: str | None = None) -> bool:
    """
    Check if any tool was actually used in the trajectory.
    Only checks trajectory for actual tool calls, not raw_pred text (which may just be following prompt format).
    
    Args:
        trajectory: List of messages or dict with "messages" key
        raw_pred: Raw prediction string (ignored - models may output "Action:" without actually calling tools)
    
    Returns:
        True if any tool was actually used (tool_calls in trajectory or tool role messages), False otherwise
    """
    try:
        # Extract messages from trajectory
        msgs = trajectory.get("messages", []) if isinstance(trajectory, dict) else trajectory
        if not isinstance(msgs, list):
            msgs = []
        
        # Check trajectory for actual tool usage
        for m in msgs:
            # Check if assistant message has tool_calls (non-empty list)
            if m.get("role") == "assistant":
                tool_calls = m.get("tool_calls")
                # tool_calls can be a list or None/empty
                if tool_calls and len(tool_calls) > 0:
                    return True
            # Check if there are tool role messages (tool observations - actual tool execution results)
            if m.get("role") == "tool":
                return True
        
        # Do NOT check raw_pred for "Action:" patterns - models may output this without actually calling tools
        return False
    except Exception:
        # If any error occurs, return False (fail-safe)
        return False
