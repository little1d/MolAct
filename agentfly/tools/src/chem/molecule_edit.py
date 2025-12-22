from agentfly.tools import tool
from rdkit import Chem
from rdkit.Chem import AllChem

def _get_valid_mol(smiles: str):
    """Utility to parse and validate SMILES."""
    if not smiles:
        return None, "Error: Input SMILES is empty."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Error: Invalid input SMILES."
    return mol, None

def _sanitize_and_return(new_mol, action_desc: str):
    """Utility to sanitize molecule and return formatted result."""
    # 碎片清理 (Keep Largest Fragment)
    try:
        frags = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            new_mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            action_desc += " (and cleaned up disconnected fragments)"
    except Exception:
        pass

    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        return f"Error: The modified molecule is chemically invalid ({e}). Try a different modification."
    
    new_smiles = Chem.MolToSmiles(new_mol)
    return {
        "observation": f"Success: {action_desc}. New SMILES: {new_smiles}",
        "status": "success",
        "smiles": new_smiles
    }

@tool(name="chem_add_group", description="Add a functional group to the molecule.")
def add_group(smiles: str, group_smiles: str, position_smarts: str = ""):
    """
    Add a functional group to the molecule.
    
    Args:
        smiles (str): Input molecule SMILES.
        group_smiles (str): SMILES of the group to add (e.g., 'O', 'N', 'C#N', 'C(=O)O', '[N+](=O)[O-]').
        position_smarts (str): Optional SMARTS pattern to specify where to attach the group.

    Returns:
        Dict with keys like "smiles"/"observation" on success, or an error string on failure.
    """
    mol, err = _get_valid_mol(smiles)
    if err: return err
    
    # 修正一些常见的非标准写法
    group_smiles = group_smiles.lstrip("-")
    if group_smiles == "NO2" or group_smiles == "N=N-O": group_smiles = "[N+](=O)[O-]"
    if group_smiles == "CHO": group_smiles = "C=O"
    if group_smiles == "OH": group_smiles = "O"
    if group_smiles == "NH2": group_smiles = "N"
    if group_smiles == "NC#N": group_smiles = "C#N"
    if group_smiles == "C(=O)O": group_smiles = "C(=O)[OH]" # Explicit hydroxyl often safer
    
    add_mol = Chem.MolFromSmiles(group_smiles)
    if add_mol is None:
        return f"Error: Invalid group_smiles: {group_smiles}. Examples: 'O' (hydroxyl), 'N' (amine), '[N+](=O)[O-]' (nitro)."

    # 确定添加位置
    target_atom_idx = None
    if position_smarts:
        patt = Chem.MolFromSmarts(position_smarts)
        if patt is None:
            return f"Error: Invalid position_smarts: {position_smarts}"
        if not mol.HasSubstructMatch(patt):
            return f"Error: Position pattern '{position_smarts}' not found in the molecule."
        match = mol.GetSubstructMatch(patt)
        target_atom_idx = match[0]
    else:
        # 默认策略：找一个合适的碳原子（优先选sp3碳）
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                target_atom_idx = atom.GetIdx()
                break
        if target_atom_idx is None:
            for atom in mol.GetAtoms():
                if atom.GetTotalNumHs() > 0:
                    target_atom_idx = atom.GetIdx()
                    break
        
        if target_atom_idx is None:
            return "Error: Could not find a suitable position to add the group. Try specifying 'position_smarts'."

    # 执行添加
    try:
        combined = Chem.CombineMols(mol, add_mol)
        em = Chem.EditableMol(combined)
        add_atom_idx = mol.GetNumAtoms()  # 添加片段的起始索引
        
        # 尝试添加单键
        em.AddBond(target_atom_idx, add_atom_idx, Chem.BondType.SINGLE)
        new_mol = em.GetMol()
        
        return _sanitize_and_return(new_mol, f"added '{group_smiles}'")
    except Exception as e:
        return f"Error adding group: {str(e)}"

@tool(name="chem_remove_group", description="Remove a substructure from the molecule.")
def remove_group(smiles: str, remove_smarts: str):
    """
    Remove a substructure from the molecule.
    
    Args:
        smiles (str): Input molecule SMILES.
        remove_smarts (str): SMARTS pattern of the substructure to remove (e.g., '[Cl]', '[N;H2]').

    Returns:
        Dict with keys like "smiles"/"observation" on success, or an error string on failure.
    """
    mol, err = _get_valid_mol(smiles)
    if err: return err
    
    patt = Chem.MolFromSmarts(remove_smarts)
    if patt is None:
        return f"Error: Invalid SMARTS pattern: {remove_smarts}"

    if not mol.HasSubstructMatch(patt):
        return f"Error: Pattern '{remove_smarts}' not found in the molecule."

    try:
        match = mol.GetSubstructMatch(patt)
        if match:
            em = Chem.EditableMol(mol)
            for idx in sorted(match, reverse=True):
                em.RemoveAtom(int(idx))
            new_mol = em.GetMol()
            return _sanitize_and_return(new_mol, f"removed '{remove_smarts}'")
        else:
            return "Error: Removal failed."
    except Exception as e:
        return f"Error removing group: {str(e)}"

@tool(name="chem_replace_group", description="Replace a substructure with another group.")
def replace_group(smiles: str, old_smarts: str, new_smiles: str):
    """
    Replace a substructure with another group.
    
    Args:
        smiles (str): Input molecule SMILES.
        old_smarts (str): SMARTS pattern of the substructure to replace.
        new_smiles (str): SMILES of the new group to insert.

    Returns:
        Dict with keys like "smiles"/"observation" on success, or an error string on failure.
    """
    mol, err = _get_valid_mol(smiles)
    if err: return err

    patt = Chem.MolFromSmarts(old_smarts)
    if patt is None:
        return f"Error: Invalid old_smarts: {old_smarts}"

    if not mol.HasSubstructMatch(patt):
        return f"Error: Pattern '{old_smarts}' not found in the molecule."

    # 修正一些常见的非标准写法
    new_smiles = new_smiles.lstrip("-")
    if new_smiles == "NO2" or new_smiles == "N=N-O": new_smiles = "[N+](=O)[O-]"
    if new_smiles == "CHO": new_smiles = "C=O"
    if new_smiles == "OH": new_smiles = "O"
    if new_smiles == "NH2": new_smiles = "N"
    if new_smiles == "NC#N": new_smiles = "C#N"
    if new_smiles == "C(=O)O": new_smiles = "C(=O)[OH]"

    repl = Chem.MolFromSmiles(new_smiles)
    if repl is None:
        return f"Error: Invalid new_smiles: {new_smiles}"

    try:
        # 使用 ReplaceSubstructs (只替换第一个匹配项)
        res = Chem.ReplaceSubstructs(mol, patt, repl, replaceAll=False)
        if res:
            new_mol = res[0]
            return _sanitize_and_return(new_mol, f"replaced '{old_smarts}' with '{new_smiles}'")
        else:
            return "Error: Replacement failed due to structural constraints."
    except Exception as e:
        return f"Error replacing group: {str(e)}"
