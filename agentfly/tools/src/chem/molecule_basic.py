from agentfly.tools import tool
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings for invalid SMILES
from rdkit.Chem import Crippen, QED, rdMolDescriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


@tool(name="chem_mol_validate", description="Validate a SMILES string and get basic molecular info.")
def mol_validate(smiles: str):
    """
    Validate a SMILES string and return basic molecular information.
    """
    if not smiles or not isinstance(smiles, str):
        return {"observation": "invalid: empty or non-string input", "valid": 0}
    
    smiles = smiles.strip()
    if not smiles:
        return {"observation": "invalid: empty SMILES", "valid": 0}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        
        canonical = Chem.MolToSmiles(mol)
        num_atoms = mol.GetNumHeavyAtoms()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        
        return {
            "observation": "ok",
            "valid": 1,
            "canonical": canonical,
            "num_atoms": num_atoms,
            "molecular_formula": formula
        }
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_calc_properties", description="Calculate molecular properties and optional similarity to a reference molecule.")
def calc_properties(smiles: str, ref_smiles: str = None):
    """
    Compute common molecular properties and optional Tanimoto similarity
    to a reference molecule.
    """
    if not smiles or not isinstance(smiles, str):
        return {"observation": "invalid: empty or non-string input", "valid": 0}
    
    smiles = smiles.strip()
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}

        # Basic properties
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        qed_score = QED.qed(mol)
        
        # Synthetic accessibility
        try:
            sa = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
        except (AttributeError, Exception):
            sa = None
        
        # Additional drug-like properties
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        
        # Similarity calculation
        sim = None
        if ref_smiles:
            ref_smiles = ref_smiles.strip()
            ref = Chem.MolFromSmiles(ref_smiles)
            if ref:
                try:
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(ref, 2, nBits=2048)
                    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                except Exception:
                    sim = None

        result = {
            "observation": "ok",
            "valid": 1,
            "mw": round(float(mw), 2),
            "logp": round(float(logp), 2),
            "qed": round(float(qed_score), 3),
            "sa": round(float(sa), 2) if sa is not None else None,
            "hbd": int(hbd),
            "hba": int(hba),
            "rotatable_bonds": int(rotatable),
            "tpsa": round(float(tpsa), 2),
        }
        
        if sim is not None:
            result["sim"] = round(float(sim), 4)
        
        return result
        
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_compare_molecules", description="Compare two molecules and check if they are the same or different.")
def compare_molecules(smiles1: str, smiles2: str):
    """
    Compare two molecules to check if they are identical (same canonical SMILES).
    """
    if not smiles1 or not smiles2:
        return {"observation": "Error: Both SMILES must be provided", "identical": False}
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1.strip())
        mol2 = Chem.MolFromSmiles(smiles2.strip())
        
        if mol1 is None:
            return {"observation": f"Error: Invalid first SMILES: {smiles1}", "identical": False}
        if mol2 is None:
            return {"observation": f"Error: Invalid second SMILES: {smiles2}", "identical": False}
        
        canonical1 = Chem.MolToSmiles(mol1)
        canonical2 = Chem.MolToSmiles(mol2)
        
        identical = (canonical1 == canonical2)
        
        # Calculate similarity
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        if identical:
            obs = "The molecules are identical."
        else:
            obs = f"The molecules are different. Similarity: {sim:.2%}"
        
        return {
            "observation": obs,
            "identical": identical,
            "sim": round(float(sim), 4),
            "canonical1": canonical1,
            "canonical2": canonical2
        }
        
    except Exception as e:
        return {"observation": f"Error: {str(e)}", "identical": False}


# --- Property tools for mol optimization ---
def _esol(smiles: str):
    """
    ESOL (Delaney) logS estimator used by many benchmarks.
    logS = 0.16 - 1.5 - 0.012*MW - logP - 0.066*RB + 0.066*AP
    MW: 分子量 molecular weight
    logP: 脂水分配系数 Crippen logP
    RB: 可旋转键数量 Rotatable Bonds
    AP: 芳香原子比例 Aromatic Proportion = 芳香原子数 / 重原子个数
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = aromatic / heavy if heavy > 0 else 0.0
    logS = 0.16 - 1.5 - 0.012 * mw - logp - 0.066 * rb + 0.066 * ap
    return float(logS)


@tool(name="chem_calc_logp", description="Calculate LogP using RDKit (Morgan radius=2, nBits=2048).")
def calc_logp(smiles: str):
    """Calculate Crippen logP for a molecule.

    Args:
        smiles (str): Input SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        logp = Crippen.MolLogP(mol)
        return {"observation": "ok", "valid": 1, "value": float(logp)}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_calc_solubility", description="Calculate ESOL LogS (higher is more soluble).")
def calc_solubility(smiles: str):
    """Predict ESOL aqueous solubility (logS).

    Args:
        smiles (str): Input SMILES.
    """
    try:
        val = _esol(smiles)
        if val is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        return {"observation": "ok", "valid": 1, "value": float(val)}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_calc_qed", description="Calculate QED drug-likeness score.")
def calc_qed(smiles: str):
    """Compute RDKit QED drug-likeness score.

    Args:
        smiles (str): Input SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        qed_score = QED.qed(mol)
        return {"observation": "ok", "valid": 1, "value": float(qed_score)}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_tanimoto_similarity", description="Tanimoto similarity using Morgan fingerprint (radius=2, nBits=2048).")
def tanimoto_similarity(smiles_a: str, smiles_b: str):
    """Compute Tanimoto similarity of two SMILES using Morgan fingerprints.

    Args:
        smiles_a (str): SMILES of molecule A.
        smiles_b (str): SMILES of molecule B.
    """
    try:
        ma = Chem.MolFromSmiles(smiles_a) if smiles_a else None
        mb = Chem.MolFromSmiles(smiles_b) if smiles_b else None
        if ma is None or mb is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        fp1 = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return {"observation": "ok", "valid": 1, "similarity": float(sim)}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_murcko_scaffold", description="Extract Bemis–Murcko scaffold as SMILES.")
def murcko_scaffold(smiles: str):
    """Extract the Bemis–Murcko scaffold of a molecule.

    Args:
        smiles (str): Input SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold is not None else ""
        return {"observation": "ok", "valid": 1, "scaffold": scaffold_smiles}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}


@tool(name="chem_scaffold_similarity", description="Tanimoto similarity between Murcko scaffolds (Morgan radius=2, nBits=2048).")
def scaffold_similarity(smiles_a: str, smiles_b: str):
    """Compare two molecules by Bemis–Murcko scaffold Tanimoto similarity.

    Args:
        smiles_a (str): SMILES of molecule A.
        smiles_b (str): SMILES of molecule B.
    """
    try:
        ma = Chem.MolFromSmiles(smiles_a) if smiles_a else None
        mb = Chem.MolFromSmiles(smiles_b) if smiles_b else None
        if ma is None or mb is None:
            return {"observation": "invalid: cannot parse SMILES", "valid": 0}
        sa = MurckoScaffold.GetScaffoldForMol(ma)
        sb = MurckoScaffold.GetScaffoldForMol(mb)
        if sa is None or sb is None:
            return {"observation": "invalid: empty scaffold", "valid": 0}
        # 与 ChemCoTBench 评测保持一致：radius=2, nBits=1024
        fp1 = AllChem.GetMorganFingerprintAsBitVect(sa, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(sb, 2, nBits=1024)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        return {"observation": "ok", "valid": 1, "similarity": float(sim)}
    except Exception as e:
        return {"observation": f"invalid: {str(e)}", "valid": 0}
