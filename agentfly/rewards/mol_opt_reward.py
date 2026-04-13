from agentfly.rewards import reward
from agentfly.tools import TOOL_REGISTRY
from rdkit import Chem
import json
import os
import fcntl
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict

# Reuse the SMILES extractor / tool detection from mol_edit
from agentfly.rewards.mol_edit_reward import extract_last_valid_smiles, _check_tool_used

# Trajectory logging setup
# Use smaller stride for debugging, can be increased later for production
LOG_STRIDE = 5000  # log roughly every LOG_STRIDE calls
_log_counter = 0
_chain_collector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_expected_chains_per_group = 4  # default num_chains
# Support environment variable for log file path (useful for running multiple experiments)
# Use absolute path to avoid issues with working directory
_log_file_path = os.environ.get("MOL_OPT_TRAJ_FILE", "logs/mol_opt_traj.jsonl")
LOG_FILE = Path(_log_file_path) if os.path.isabs(_log_file_path) else Path(_log_file_path).resolve()

PROP_WEIGHT = 0.8
SCAFFOLD_WEIGHT = 0.15
TOOL_BONUS = 0.05
# ChemCoTBench 阈值：logp/solubility 用 0.5，其余 0.3
THRESH_DICT = {
    "logp": 0.5,
    "solubility": 0.5,
    "qed": 0.3,
    # Support both "drd"/"drd2" for compatibility
    "drd": 0.3,
    "drd2": 0.3,
    # Support both "jnk"/"jnk3" for compatibility
    "jnk": 0.3,
    "jnk3": 0.3,
    # Support both "gsk"/"gsk3b" for compatibility
    "gsk": 0.3,
    "gsk3b": 0.3,
}


async def _call_tool(name: str, **kwargs):
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"valid": 0}
    res = tool(**kwargs)
    if hasattr(res, "__await__"):
        res = await res
    return res


def _get_prop_value(res):
    """
    Extract property value from tool response.
    Tool responses can be in two formats:
    1. Direct format: {'valid': 1, 'value': ...} or {'valid': 1, 'score': ...}
    2. Wrapped format: {'info': {'valid': 1, 'value': ...}} or {'info': {'valid': 1, 'score': ...}}
    """
    if not isinstance(res, dict):
        return None
    
    # Check if value is wrapped in 'info' field (from tool registry wrapper)
    if "info" in res and isinstance(res["info"], dict):
        info = res["info"]
        if info.get("valid") != 1:
            return None
        return info.get("value") or info.get("score")
    
    # Direct format
    if res.get("valid") != 1:
        return None
    return res.get("value") or res.get("score")


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _prop_tool_for(subtask: str):
    subtask = (subtask or "").lower()
    mapping = {
        "logp": ("chem_calc_logp", None),
        "solubility": ("chem_calc_solubility", None),
        "qed": ("chem_calc_qed", None),
        # Support both "drd"/"drd2" for compatibility
        "drd": ("chem_oracle_score", "drd"),
        "drd2": ("chem_oracle_score", "drd"),
        # Support both "jnk"/"jnk3" for compatibility
        "jnk": ("chem_oracle_score", "jnk"),
        "jnk3": ("chem_oracle_score", "jnk"),
        # Support both "gsk"/"gsk3b" for compatibility
        "gsk": ("chem_oracle_score", "gsk"),
        "gsk3b": ("chem_oracle_score", "gsk"),
    }
    return mapping.get(subtask)


def _extract_group_id(trajectory: Any, **kwargs) -> Optional[str]:
    """Extract group_id from trajectory or kwargs for grouping chains."""
    if isinstance(trajectory, dict):
        meta = trajectory.get("meta") or {}
        if "group_id" in meta:
            return meta["group_id"]
        msgs = trajectory.get("messages", [])
        if msgs and isinstance(msgs[0], dict):
            meta0 = msgs[0].get("meta")
            if meta0 and "group_id" in meta0:
                return meta0["group_id"]
    return kwargs.get("group_id") or kwargs.get("uid") or "default_group"


def _log_chain_group(group_id: str, chains: List[Dict[str, Any]], ref_smiles: Optional[str] = None):
    _dump_group_to_file(group_id, chains, ref_smiles)


def _dump_group_to_file(group_id: str, chains: List[Dict[str, Any]], ref_smiles: Optional[str]):
    """Save trajectory group to file."""
    # Check if trajectory logging is disabled
    if os.environ.get("DISABLE_TRAJ_LOG", "0") == "1":
        return
    
    try:
        # Ensure LOG_FILE uses absolute path
        log_file_abs = LOG_FILE.absolute() if not LOG_FILE.is_absolute() else LOG_FILE
        # Ensure parent directory exists and is writable
        log_file_abs.parent.mkdir(parents=True, exist_ok=True)
        
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
                    "prop_score": info.get("prop_score"),
                    "scaffold_sim": info.get("scaffold_sim"),
                    "tool_bonus": info.get("tool_bonus"),
                    "extracted": info.get("extracted"),
                    "raw_pred": info.get("raw_pred"),
                    "trajectory": info.get("trajectory"),
                    "debug_info": info.get("debug_info"),
                }
            )
        # Use file lock to prevent concurrent write issues in multi-process environment
        with log_file_abs.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
    except Exception as e:
        print(f"[mol_opt_reward] failed to write log file {LOG_FILE} (abs: {LOG_FILE.absolute() if hasattr(LOG_FILE, 'absolute') else LOG_FILE}): {e}", flush=True)
        import traceback
        traceback.print_exc()


@reward(name="mol_opt_reward")
async def mol_opt_reward(
    prediction: str,
    trajectory,
    ref_smiles: str = None,
    num_chains: int = None,
    **kwargs,
):
    """
    分子优化奖励：
      - 非法 SMILES：-1
      - 合法：0.8*属性提升 + 0.15*骨架相似 + 0.05*工具奖励
        属性提升采用 (prop_pred - prop_src)/THRESH_DEFAULT 裁剪到 [0,1]
        骨架相似使用 Murcko scaffold + Morgan Tanimoto (半径2, 2048 bits)
    """
    global _log_counter, _chain_collector, _expected_chains_per_group
    if num_chains:
        _expected_chains_per_group = num_chains
    _log_counter += 1
    # Log every LOG_STRIDE calls (LOG_STRIDE=1 means log every call)
    do_log = (_log_counter % LOG_STRIDE == 0)
    
    # 默认输出保持 batch 对齐
    # Note: extracted, raw_pred, trajectory are only used for logging, not for training
    # They should not be included in the return dict to avoid numpy array conversion issues
    result = {
        "reward": -1.0,
        "valid": 0,
        "prop_score": 0.0,
        "scaffold_sim": 0.0,
        "tool_bonus": 0.0,
        "debug_info": "",
    }
    
    # Store logging fields separately (will be used in _log_chain_group)
    _logging_fields = {
        "extracted": None,
        "raw_pred": prediction,
        "trajectory": trajectory,
    }

    # 解包 extra_info（兼容 dataset 的包装方式）
    extra_info = kwargs.get("extra_info", {})
    if isinstance(extra_info, (list, tuple)) and extra_info and isinstance(extra_info[0], dict):
        extra_info = extra_info[0]
    if isinstance(extra_info, dict):
        for k in ["task", "subtask", "src_smiles", "ref_smiles"]:
            if k not in kwargs and k in extra_info:
                kwargs[k] = extra_info[k]

    task = (kwargs.get("task") or "").lower()
    subtask = (kwargs.get("subtask") or "").lower()
    src_smiles = kwargs.get("src_smiles") or kwargs.get("source_smiles")

    try:
        pred_smiles = extract_last_valid_smiles(prediction, trajectory=trajectory)
    except Exception:
        pred_smiles = None
    
    # Update logging fields
    _logging_fields["extracted"] = pred_smiles

    # 基础合法性
    if not pred_smiles:
        result["debug_info"] = "no_smiles_extracted"
        _logging_fields["extracted"] = None
    else:
        try:
            if Chem.MolFromSmiles(pred_smiles) is None:
                result["debug_info"] = "invalid_pred"
            else:
                # 处理 src_smiles 缺失的情况
                if not src_smiles or Chem.MolFromSmiles(src_smiles) is None:
                    result["debug_info"] = "invalid_src"
                else:
                    # 选择属性工具
                    tool_name, prop_arg = _prop_tool_for(subtask) or (None, None)
                    if not tool_name:
                        result["debug_info"] = f"unsupported_subtask:{subtask}"
                    else:
                        # 计算属性值
                        src_res = await _call_tool(tool_name, smiles=src_smiles, prop=prop_arg) if prop_arg else await _call_tool(tool_name, smiles=src_smiles)
                        pred_res = await _call_tool(tool_name, smiles=pred_smiles, prop=prop_arg) if prop_arg else await _call_tool(tool_name, smiles=pred_smiles)
                        src_val = _get_prop_value(src_res)
                        pred_val = _get_prop_value(pred_res)

                        if src_val is None or pred_val is None:
                            result["debug_info"] = "prop_calc_failed"
                        else:
                            # 属性提升分
                            delta = pred_val - src_val
                            thresh = THRESH_DICT.get(subtask, 0.3)
                            prop_score = _clip01(delta / thresh)

                            # 骨架相似
                            scaff_res = await _call_tool("chem_scaffold_similarity", smiles_a=src_smiles, smiles_b=pred_smiles)
                            # Extract similarity from tool response (may be wrapped in 'info' field)
                            if isinstance(scaff_res, dict):
                                if "info" in scaff_res and isinstance(scaff_res["info"], dict):
                                    scaffold_sim = scaff_res["info"].get("similarity", 0.0)
                                else:
                                    scaffold_sim = scaff_res.get("similarity", 0.0)
                            else:
                                scaffold_sim = 0.0

                            used_tool = _check_tool_used(trajectory, prediction)
                            tool_bonus = TOOL_BONUS if used_tool else 0.0

                            reward_val = PROP_WEIGHT * prop_score + SCAFFOLD_WEIGHT * float(scaffold_sim or 0.0) + tool_bonus
                            reward_val = max(-1.0, min(1.0, reward_val))

                            result.update(
                                {
                                    "reward": float(reward_val),
                                    "valid": 1,
                                    "prop_score": float(prop_score),
                                    "scaffold_sim": float(scaffold_sim or 0.0),
                                    "tool_bonus": float(tool_bonus),
                                    "debug_info": f"task={task}, subtask={subtask}, src={src_smiles}, delta={delta:.4f}, prop={pred_val:.4f}/{src_val:.4f}",
                                }
                            )
        except Exception as e:
            result["debug_info"] = f"Exception: {str(e)}"
    
    # Always add to chain collector for logging (similar to mol_edit_reward)
    # This ensures all chains (successful or failed) are logged
    group_id = _extract_group_id(trajectory, **kwargs) or f"unknown_{_log_counter}"
    # Combine result with logging fields for trajectory logging
    log_entry = {**result, **_logging_fields}
    _chain_collector[group_id].append(log_entry)
    
    # Save trajectory: log once per group when group is complete; save all chains for the group
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
