# Oracle Files

This directory contains oracle files used for protein activation scoring in molecular optimization tasks (DRD-2, JNK-3, GSK-3β).

## Automatic Download

When running inference or evaluation scripts from the root directory, the TDC (Therapeutics Data Commons) library will automatically download oracle files to this directory if they are not present.

## Manual Download

You can also manually download the oracle files in advance. The required files are:

- `drd2_current.pkl` - DRD-2 protein activation oracle
- `jnk3_current.pkl` - JNK-3 protein activation oracle  
- `gsk3b_current.pkl` - GSK-3β protein activation oracle
- `fpscores.pkl` - Fingerprint scores for oracle calculations

These files are used by the `oracle_score` tool during molecular optimization tasks.