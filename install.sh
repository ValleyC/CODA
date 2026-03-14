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

echo "[CODA] Installing runtime dependencies required by mamba-ssm..."
python -m pip install einops==0.8.2

echo "[CODA] Building causal-conv1d from source..."
python -m pip install --no-cache-dir --no-build-isolation causal-conv1d==1.2.2.post1

echo "[CODA] Building mamba-ssm from source against the active PyTorch..."
python -m pip install --no-cache-dir --no-build-isolation --no-deps mamba-ssm==2.2.2

echo "[CODA] Installing CODA in editable mode..."
python -m pip install -e .

echo "[CODA] Verifying key imports..."
python - <<'PY'
import torch
import madmom
from mamba_ssm.modules.mamba_simple import Mamba

print("torch:", torch.__version__)
print("mamba layer:", Mamba.__name__)
print("madmom:", getattr(madmom, "__version__", "unknown"))
PY

echo "[CODA] Installation complete."
