#!/usr/bin/env bash
# CODA: Full reproduction script
# This script downloads data, trains the model, and evaluates it.

set -Eeuo pipefail

# ============================================================
# 1. Environment setup
# ============================================================
echo "=== Setting up environment ==="
CONDA_BASE=$(conda info --base)
# `conda activate` is a shell function and is unavailable in many clean,
# non-interactive shells until this hook is sourced explicitly.
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx coda; then
    conda env update --name coda --file environment.yml
else
    conda env create --file environment.yml
fi
conda activate coda
bash install.sh

# ============================================================
# 2. Data preparation
# ============================================================
echo "=== Downloading MSMD dataset ==="
mkdir -p data
cd data
if [ ! -d "msmd" ]; then
    wget https://zenodo.org/record/4745838/files/msmd.zip
    unzip msmd.zip
    rm msmd.zip
fi
cd ..

# ============================================================
# 3. Full training and evaluation
# ============================================================
# This uses only msmd_train for optimization and msmd_valid for model
# selection. It produces atomic resume checkpoints, regenerates the fixed
# 66-piece annotated and 28-piece random jump sets in clean directories, and
# evaluates msmd_test only after both training phases are complete.
bash scripts/run_full_pipeline.sh
