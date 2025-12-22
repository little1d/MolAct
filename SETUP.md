# Setup Guide for MolAct

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MolAct.git
cd MolAct
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For chemistry tools, you may need to install RDKit separately:
```bash
pip install rdkit-pypi
```

### 3. Download Models

Models are available on Hugging Face. You can download them using:

```bash
# Using huggingface-cli
huggingface-cli download little1d/MolEditAgent-3B --local-dir ./models/MolEditAgent-3B
huggingface-cli download little1d/MolOptAgent-3B --local-dir ./models/MolOptAgent-3B
```

Or use them directly from Hugging Face in the inference scripts (they will be downloaded automatically).

### 4. Verify Installation

```bash
python -c "from agentfly.agents.react.react_agent import ReactAgent; print('Installation successful!')"
```

## Quick Test

Run a simple inference test:

```bash
python scripts/run_mol_edit_agent.py \
    --model_path little1d/MolEditAgent-3B \
    --input_file ChemCoTBench_benchmark/chemcotbench/mol_edit/add.json \
    --output_file test_output.json \
    --max_turns 4
```

## Troubleshooting

### Issue: RDKit installation fails
- Try: `conda install -c conda-forge rdkit`
- Or use: `pip install rdkit-pypi`

### Issue: CUDA out of memory
- Use smaller models (3B instead of 7B)
- Reduce `max_new_tokens` parameter
- Use CPU inference (slower but works)

### Issue: Import errors
- Make sure you're in the repository root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

