# MolAct: Molecular Action Agents for Chemistry Tasks

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/little1d/MolAct)

MolAct is a framework for molecular chemistry tasks using large language models with tool-augmented reasoning. This repository contains the inference and evaluation code for MolEditAgent and MolOptAgent models.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

#### Molecular Editing (MolEdit)

```bash
python scripts/run_mol_edit_agent.py \
    --model_path little1d/MolEditAgent-3B \
    --input_file ChemCoTBench_benchmark/chemcotbench/mol_edit/add.json \
    --output_file outputs/mol_edit_results.json \
    --backend transformers \
    --max_turns 16
```

#### Molecular Optimization (MolOpt)

```bash
python scripts/run_mol_opt_agent.py \
    --model_path little1d/MolOptAgent-3B \
    --input_file ChemCoTBench_benchmark/chemcotbench/mol_opt/logp.json \
    --output_file outputs/mol_opt_results.json \
    --max_turns 16
```

### Evaluation

See `ChemCoTBench/` directory for evaluation framework and scripts.

## 📦 Models and Data

All models and training data are available on Hugging Face:

- **Collection**: [little1d/MolAct](https://huggingface.co/collections/little1d/MolAct)
- **Models**:
  - [MolEditAgent-3B](https://huggingface.co/little1d/MolEditAgent-3B)
  - [MolEditAgent-7B](https://huggingface.co/little1d/MolEditAgent-7B)
  - [MolOptAgent-3B](https://huggingface.co/little1d/MolOptAgent-3B)
  - [MolOptAgent-7B](https://huggingface.co/little1d/MolOptAgent-7B)
- **Datasets**:
  - [mol_edit_data](https://huggingface.co/datasets/little1d/mol_edit_data)
  - [mol_opt_data](https://huggingface.co/datasets/little1d/mol_opt_data)

## 🛠️ Tools

The agents use chemistry tools for molecular manipulation and property calculation:
- Molecular validation and basic operations
- Property calculation (logP, QED, solubility, etc.)
- Molecular editing (add/remove/replace groups)
- Scaffold analysis and similarity metrics
- Oracle scoring for optimization tasks

## 📊 Benchmark

We evaluate on [ChemCoTBench](https://github.com/your-repo/ChemCoTBench) for both molecular editing and optimization tasks.

## 📝 Citation

If you use MolAct in your research, please cite:

```bibtex
@article{molact2024,
  title={MolAct: Molecular Action Agents for Chemistry Tasks},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

See [LICENSE](LICENSE) file for details.

## ⚠️ Note

This repository contains inference and evaluation code only. Training code is not included. For training-related inquiries, please contact the authors.

