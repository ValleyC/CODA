#!/bin/bash
# CODA: Full reproduction script
# This script downloads data, trains the model, and evaluates it.

set -e

# ============================================================
# 1. Environment setup
# ============================================================
echo "=== Setting up environment ==="
conda create -n coda python=3.10 -y
conda activate coda
pip install -r requirements.txt
pip install -e .

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
# 3. Training Phase 1: Ground-truth routing
# ============================================================
echo "=== Training Phase 1 ==="
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --tag coda_phase1 \
    --temporal_priors \
    --augment \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 5e-4 \
    --num_workers 4

# Find Phase 1 best checkpoint
PHASE1_DIR=$(ls -td params/*coda_phase1* | head -1)
echo "Phase 1 best model: ${PHASE1_DIR}/best_model.pt"

# ============================================================
# 4. Training Phase 2: Scheduled sampling
# ============================================================
echo "=== Training Phase 2 ==="
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --param_path ${PHASE1_DIR}/best_model.pt \
    --tag coda_phase2 \
    --temporal_priors \
    --augment \
    --scheduled_sampling \
    --ss_max_p 0.7 \
    --ss_ramp_epochs 5 \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-4 \
    --num_workers 4

PHASE2_DIR=$(ls -td params/*coda_phase2* | head -1)
echo "Phase 2 best model: ${PHASE2_DIR}/best_model.pt"

# ============================================================
# 5. Evaluation
# ============================================================
echo "=== Evaluating on test set ==="
for piece in data/msmd/msmd_test/*.npz; do
    piece_name=$(basename "$piece" .npz)
    echo "Evaluating: ${piece_name}"
    python scripts/evaluate.py \
        --param_path ${PHASE2_DIR}/best_model.pt \
        --test_dir data/msmd/msmd_test \
        --test_piece ${piece_name} \
        --output_dir results/
done

# ============================================================
# 6. Jump Recovery Evaluation
# ============================================================
echo "=== Generating jump-augmented test data ==="
python scripts/generate_jump_data.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --num_variants 3 \
    --seed 42

echo "=== Evaluating jump recovery ==="
for piece in data/msmd/msmd_test_jump/*.npz; do
    piece_name=$(basename "$piece" .npz)
    echo "Evaluating (jump): ${piece_name}"
    python scripts/evaluate.py \
        --param_path ${PHASE2_DIR}/best_model.pt \
        --test_dir data/msmd/msmd_test_jump \
        --test_piece ${piece_name} \
        --break_mode \
        --output_dir results/jump/
done

echo "=== Done! Results saved to results/ ==="
