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
conda env create -f environment.yml
conda activate coda
bash install.sh
```

> **Tested setup:** Linux, Python 3.10, CUDA 12.1, PyTorch 2.2.0, `mamba-ssm` 2.2.2, `madmom` 0.17.dev0.
>
> **Why `install.sh` is required:** `mamba-ssm` prebuilt wheels often mismatch the local PyTorch ABI. The script builds `mamba-ssm` from source against the active PyTorch installation.
> CODA also pins a compatible `transformers` version because `mamba-ssm` imports it at package initialization time, even though CODA only uses the Mamba backbone.
>
> **Why `madmom` comes from Git:** the PyPI release is not compatible with Python 3.10. CODA installs the current development version from the official repository.
>
> **Why `numpy==1.26.4` is pinned:** `madmom` currently fails with NumPy 2.x because its compiled extensions target the NumPy 1.x ABI.
>
> **System prerequisites:** a working C/C++ toolchain is required to build `mamba-ssm`. `ffmpeg` is included in `environment.yml` because CODA uses it for video export.

If you need Weights & Biases logging, install it separately:

```bash
pip install wandb
```

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

### Standard Tracking (Single Piece)

```bash
python scripts/evaluate.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName
```

This prints onset error ratios at multiple thresholds, system/bar accuracy, and frame-level pixel error.

### Jump Recovery (Single Piece)

Evaluate on the repeat-aware jump-augmented test set with break mode enabled:
```bash
python scripts/evaluate.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --test_piece PieceName \
    --break_mode
```

This prints per-piece jump recovery metrics (system recovery rate, latency, post-jump tracking accuracy) at multiple thresholds.

### Batch Evaluation (All Pieces)

Evaluate all pieces in a directory and aggregate metrics:
```bash
# Standard tracking — Table 1 (MSMD test set, 94 pieces)
python scripts/evaluate_batch.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --label "CODA (ours)" \
    --metrics_dir results/metrics/standard \
    --save_summary results/standard_summary.json

# Jump recovery — Table 2, repeat subset
python scripts/evaluate_batch.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --break_mode \
    --label "CODA (full) - repeat" \
    --metrics_dir results/metrics/repeat \
    --save_summary results/repeat_summary.json

# Jump recovery — Table 2, random subset
python scripts/evaluate_batch.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/random \
    --break_mode \
    --label "CODA (full) - random" \
    --metrics_dir results/metrics/random \
    --save_summary results/random_summary.json
```

For standard tracking, the script prints onset error ratios (<=0.05s through <=5.00s), system accuracy, bar accuracy, and a Table 1 LaTeX row. For jump recovery, it additionally prints recovery rates, latency, post-jump tracking accuracy, and a Table 2 LaTeX row.

### Video Generation

Generate a tracking visualization video for a single piece:
```bash
python scripts/evaluate.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName \
    --output_dir videos/
```

Use `--plot` to also display the visualization in a live window. Use `--no_video` to skip video generation (metrics only).

To batch-evaluate with inline video generation:
```bash
python scripts/evaluate_batch.py \
    --param_path pretrained/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --break_mode \
    --label "CODA (full) - repeat" \
    --metrics_dir results/metrics/repeat \
    --save_summary results/repeat_summary.json \
    --with_video --video_dir results/videos/repeat
```

Videos are generated inline (one piece at a time) during the metrics pass.

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
