---
license: mit
language:
- en
tags:
- chemistry
- biology
size_categories:
- 1K<n<10K
---

# Tasks
## mol_und
- fg-level
  - `fg_count.json`
    - 100 samples across 38 different functional groups detection
  - `ring_count.json`
    - 20 samples, 9 types of ring unit
- scaffold-level
  - `Murcko_scaffold.json`
    - 40 samples, using MurckoScaffold extraction
  - `ring_system_scaffold.json`
    - 60 samples, extract ring system as scaffolds
- SMILES-level
  - `equivalence.json`
    - 50 samples, each smiles -> mutate -> permutate, mutated smiles differs from original smiles
    - 50 samples, each smiles -> permutate, permutated smiles is same with original smiles

## mol_edit
- `add.json`
  - 20 samples, covers 10 func groups addition
- `delete.json`
  - 20 samples, covers 10 func groups deletion
- `sub.json`
  - 60 samples, covers 37 func groups substitution

## mol_opt
- `drd.json`
  - 100 samples, target-level
- `gsk.json`
  - 100 samples, target-level
- `jnk.json`
  - 100 samples, target-level
- `logp.json`
  - 100 samples, physicochemical-level
- `qed.json`
  - 100 samples, physicochemical-level
- `solubility.json`
  - 100 samples, physicochemical-level

## reaction
- forward reaction prediction
  - `fs.json`
    - 100 samples, 100 rxn-cls (each has one sample)
- (single-step) retrosynthesis prediction
  - `retro.json`
    - 100 samples, 100 rxn-cls (each has one sample)
- reaction condition prediction/recommendation
  - `rcr.json`
    - 90 samples. 10 types of reaction. Each type has 3 samples for 'Catalyst' prediction, 3 for 'Reagent' prediction and 3 for 'Solvent' prediction
- Next elementary step product prediction (NEPP)
  - `nepp.json`
    - given former elementary steps description, predict next elementary step's product.
    - 85 rxn cls, each has 1 sample
- Mechanism Route Selection (MechSel)
  - `mechsel.json`
    - 100 samples
   
# Links
- Our larger CoT dataset [ChemCoTBench-CoT](https://huggingface.co/datasets/OpenMol/ChemCoTBench-CoT)

# Citation
If you find our work helpful, feel free to give us a cite.
```
@article{li2025chemicalqaevaluatingllms,
      title={Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations}, 
      author={Hao Li and He Cao and Bin Feng and Yanjun Shao and Xiangru Tang and Zhiyuan Yan and Li Yuan and Yonghong Tian and Yu Li},
      year={2025},
      eprint={2505.21318},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.21318}, 
}
```