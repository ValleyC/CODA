# CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Score Following

Official implementation of "CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Score Following" (ISMIR 2026).

CODA is a real-time score following system that tracks a live audio performance on sheet music images. It formulates score tracking as a cascaded selection task over known system and bar candidates, combined with a silence-driven jump recovery mechanism for handling score discontinuities.

## Architecture

CODA processes audio through a causal Mamba state-space encoder and the score page through a convolutional backbone with FPN. Three cascaded stages progressively narrow the prediction:

1. **System Selection**: Classifies the active system among all systems on the page using ROI-aligned features with FiLM conditioning and cross-attention over the audio history.
2. **Bar Selection**: Classifies the active bar within the selected system, using the same FiLM + cross-attention pipeline with independent parameters.
3. **Note Localization**: Regresses continuous bar-local coordinates within the selected bar via FiLM-conditioned features and sigmoid output.

Beam search with learned temporal priors decodes the cascade over time. A silence-driven break mode enables recovery from arbitrary score discontinuities (repeats, D.C., coda jumps).

## Requirements

### Environment Setup

```bash
# 1. Create conda environment
conda create -n coda python=3.10
conda activate coda

# 2. Install PyTorch with CUDA (adjust cu121 to match your CUDA version)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install Mamba SSM (requires torch+CUDA)
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=1.0

# 4. Install cython (needed to build madmom from source)
pip install cython

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Install CODA package
pip install -e .
```
- tensorboard

## Data Preparation

### MSMD Dataset

We use the preprocessed version of [MSMD](https://github.com/CPJKU/msmd) provided by [Henkel & Widmer (2021)](https://github.com/CPJKU/cyolo_score_following), where each piece is stored as a `.npz` file (score images and annotations) paired with a `.wav` file (synthesized audio at 22050 Hz).

```bash
mkdir -p data
cd data
wget https://zenodo.org/record/4745838/files/msmd.zip
unzip msmd.zip
```

Expected directory structure:
```
data/
  msmd/
    msmd_train/         # 354 pieces
      PieceName.npz
      PieceName.wav
      ...
    msmd_valid/         # 19 pieces
      ...
    msmd_test/          # 94 pieces
      ...
```

### Repeat-Aware Jump-Augmented Test Set

Generate the jump-augmented test benchmark for discontinuity recovery evaluation. Pieces with written repeats (repeat barlines, da capo, etc.) follow their actual repeat structure; pieces without repeats receive random jumps.

```bash
python scripts/generate_repeat_test.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --annotations data/repeat_annotations.json
```

This produces two subsets under `msmd_test_jump/`:
- `repeat/` — pieces with real repeat structures (jumps follow annotated performance order)
- `random/` — pieces without repeats (random jumps inserted)

See `data/repeat_annotations.json` for the per-piece repeat structure annotations.

## Training

### Phase 1: Ground-Truth Routing

```bash
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --tag coda_phase1 \
    --temporal_priors \
    --augment \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 5e-4
```

### Phase 2: Scheduled Sampling

Fine-tune from Phase 1 checkpoint:
```bash
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --param_path params/<phase1_dir>/best_model.pt \
    --tag coda_phase2 \
    --temporal_priors \
    --augment \
    --scheduled_sampling \
    --ss_max_p 0.7 \
    --ss_ramp_epochs 5 \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-4
```

## Evaluation

### Standard Tracking

```bash
python scripts/evaluate.py \
    --param_path params/<phase2_dir>/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName
```

### With Break Mode (Jump Recovery)

Evaluate on the repeat-aware jump-augmented test set:
```bash
python scripts/evaluate.py \
    --param_path params/<phase2_dir>/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --test_piece PieceName \
    --break_mode
```

### Generate Video Output

```bash
python scripts/evaluate.py \
    --param_path params/<phase2_dir>/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName \
    --plot \
    --output_dir videos/
```

## Pre-trained Models

Pre-trained checkpoints will be released upon paper acceptance.

## Citation

```bibtex
@inproceedings{coda2026,
  title={CODA: Cascaded Online Discontinuity-Aware Alignment for Real-Time Score Following},
  author={Anonymous},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2026}
}
```

## Acknowledgments

- Preprocessed [MSMD](https://github.com/CPJKU/msmd) data provided by [CYOLO Score Following](https://github.com/CPJKU/cyolo_score_following) (Henkel & Widmer, 2021)
- Audio encoder based on [Mamba](https://github.com/state-spaces/mamba)
