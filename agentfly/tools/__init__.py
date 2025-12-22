# Global tool registry
TOOL_REGISTRY = {}
TOOL_FACTORY = {}

from typing import List
from .tool_base import (
    Tool, 
    hallucination_tool,
    invalid_input_tool,
    tool,
    submit_tool_call,
    submit_tool_calls
)
from .src.calculate.tools import calculator
from .src.chem.molecule_basic import (
    mol_validate,
    calc_properties,
    compare_molecules,
    calc_logp,
    calc_solubility,
    calc_qed,
    tanimoto_similarity,
    murcko_scaffold,
    scaffold_similarity,
)
from .src.chem.molecule_edit import add_group, remove_group, replace_group

# Try to import TDC oracles (optional dependency)
try:
    from .src.chem.property_oracles import oracle_score, multi_objective_score
    _has_tdc = True
except ImportError:
    _has_tdc = False


# Add explicit tools in case they weren't auto-registered
EXPLICIT_TOOLS = {
    "hallucination_tool": hallucination_tool,
    "invalid_input_tool": invalid_input_tool,
    "calculator": calculator,
    "chem_mol_validate": mol_validate,
    "chem_calc_properties": calc_properties,
    "chem_compare_molecules": compare_molecules,
    "chem_calc_logp": calc_logp,
    "chem_calc_solubility": calc_solubility,
    "chem_calc_qed": calc_qed,
    "chem_tanimoto_similarity": tanimoto_similarity,
    "chem_murcko_scaffold": murcko_scaffold,
    "chem_scaffold_similarity": scaffold_similarity,
    "chem_add_group": add_group,
    "chem_remove_group": remove_group,
    "chem_replace_group": replace_group,
}

# Add TDC oracles if available
if _has_tdc:
    EXPLICIT_TOOLS["chem_oracle_score"] = oracle_score
    EXPLICIT_TOOLS["chem_multi_objective_score"] = multi_objective_score

# Update the registry with explicit tools
for _name, _tool in EXPLICIT_TOOLS.items():
    if _name not in TOOL_REGISTRY:
        TOOL_REGISTRY[_name] = _tool

def register_tool(tool_name, tool_func):
    """
    Register a tool in the tool registry.
    
    Args:
        tool_name: The name of the tool
        tool_func: The tool function or BaseTool instance
    """
    global TOOL_REGISTRY
    TOOL_REGISTRY[tool_name] = tool_func

def get_tool_from_name(tool_name: str) -> Tool:
    """
    Get a tool instance from its name.
    """
    return TOOL_REGISTRY[tool_name]

def get_tools_from_names(tool_names: List[str]) -> List[Tool]:
    """
    Get tool instances from their names.
    
    Args:
        tool_names: List of tool names
        
    Returns:
        List of BaseTool instances
        
    Raises:
        KeyError: If a tool name is not found in the registry
    """
    return [TOOL_REGISTRY[tool_name] for tool_name in tool_names]

def list_available_tools() -> List[str]:
    """
    List all available tools.
    
    Returns:
        List of tool names
    """
    return list(TOOL_REGISTRY.keys())
