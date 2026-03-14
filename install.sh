#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "coda" ]]; then
  echo "Activate the 'coda' conda environment first:"
  echo "  conda env create -f environment.yml"
  echo "  conda activate coda"
  exit 1
fi

echo "[CODA] Upgrading build tools..."
python -m pip install --upgrade pip setuptools wheel packaging ninja

echo "[CODA] Building causal-conv1d from source..."
python -m pip install --no-cache-dir --no-build-isolation causal-conv1d==1.2.2.post1

echo "[CODA] Building mamba-ssm from source against the active PyTorch..."
python -m pip install --no-cache-dir --no-build-isolation mamba-ssm==2.2.2

echo "[CODA] Installing CODA in editable mode..."
python -m pip install -e .

echo "[CODA] Verifying key imports..."
python - <<'PY'
import torch
import mamba_ssm
import madmom

print("torch:", torch.__version__)
print("mamba_ssm:", getattr(mamba_ssm, "__version__", "unknown"))
print("madmom:", getattr(madmom, "__version__", "unknown"))
PY

echo "[CODA] Installation complete."
