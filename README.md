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

Download the [official CODA JumpBench v1.0.0 release](https://huggingface.co/datasets/ValleyC/CODA-JumpBench) from Hugging Face:
```bash
python -m pip install -U huggingface_hub
hf download ValleyC/CODA-JumpBench CODA_JumpBench_v1.0.0.zip \
    --repo-type dataset --local-dir data
unzip data/CODA_JumpBench_v1.0.0.zip -d data/msmd/
```

The archive SHA-256 is `91f1acaffd29391065fee44efb021bcad0f69004a53c7a35b628ee2b6a5364b9`.

Alternatively, generate it from the base MSMD test set:
```bash
python scripts/generate_repeat_test.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --annotations data/repeat_annotations.json \
    --seed 42 \
    --clean_output
```

This produces two subsets under `msmd_test_jump/`:
- `repeat/` — pieces with real repeat structures (jumps follow annotated performance order)
- `random/` — pieces without repeats (random jumps inserted)

See `data/repeat_annotations.json` for the per-piece repeat structure annotations.

The `repeat/` subset contains 66 pieces whose fixed annotated performance
orders define every repeat destination. The `random/` subset contains 28
non-repeat pieces with pseudo-random jumps generated from the fixed seed.
Each random piece receives a stable SHA-256-derived seed, so its artifact is
identical whether generated alone or in the full benchmark and is independent
of processing order; these per-piece seeds are recorded in the manifest.
`--clean_output` removes only previously generated `.npz`/`.wav` files in
these two output directories before a full generation pass. This prevents
stale variants from changing the benchmark while leaving the annotations and
their performance orders untouched. The manifest records both exact lists.

## Training

For the complete paper-scale run (30 Phase-1 epochs, 20 Phase-2 epochs, clean
benchmark generation, and all final evaluations), use:

```bash
bash scripts/run_full_pipeline.sh
```

The pipeline uses only `msmd_train` for optimization, uses `msmd_valid` for
checkpoint selection, and does not touch `msmd_test` until training is over.
It enables mixed precision (BF16 when supported, otherwise guarded FP16), room-response augmentation when
`data/irs/openair` is available, cold-start and jump augmentation, learned
uncertainty loss weighting, and atomic resumable checkpoints.
Each completed pipeline epoch is retained as
`checkpoint_epoch_000.pt`, `checkpoint_epoch_001.pt`, and so on, in addition
to the rolling `latest_checkpoint.pt` and selected `best_checkpoint.pt`.
These permanent checkpoints contain the model, optimizer, scheduler, scaler,
loss-weight and RNG states, plus aggregate train, validation, and streaming
metrics for that epoch. Training permutations are derived from the run seed,
phase tag, and epoch, and each stochastic dataset sample receives a stable
`(phase seed, epoch, index)` augmentation seed. Consequently, restarting with
fresh persistent workers does not silently change the resumed phase's sample
order or augmentation stream, while Phase 1 and Phase 2 retain distinct
stochastic curricula.

The pipeline runs the repository test suite before training. After all
three evaluations, it accepts only complete summaries containing exactly 94
standard, 66 annotated-repeat, and 28 random-repeat pieces. It then writes an
atomic `run_manifest.json` containing SHA-256 hashes for the selected Phase 1
and Phase 2 checkpoints, final model, jump manifest, test log, and evaluation
summaries. A failed test, missing piece, partial evaluation, or malformed jump
manifest stops the pipeline without emitting `status: complete`.

Set `AMP_DTYPE=float16` or `AMP_DTYPE=bfloat16` to override automatic
precision selection. To continue a pipeline phase in the same run directory,
set `PHASE1_RESUME_STATE` or `PHASE2_RESUME_STATE` to its structured checkpoint:

```bash
PHASE1_RESUME_STATE=/absolute/path/to/latest_checkpoint.pt \
    bash scripts/run_full_pipeline.sh
```

`failure_checkpoint.pt` is deliberately diagnostic-only: it contains the
finite state after a partially trained epoch, so replaying that artifact from
batch zero would duplicate optimizer updates. Resume from `latest_checkpoint.pt`
or a permanent `checkpoint_epoch_NNN.pt`, both of which represent a completed
epoch boundary.

Temporal evidence is represented as normalized categorical transition
distributions. System transitions distinguish `same`, `forward_1`,
`backward_1`, and `far`; bar transitions distinguish `stay`, `forward_1`,
`forward_2`, `backward_1`, and `far`. Category probability is divided among
the candidates currently in that category, so changing the number of systems
or bars does not silently change the prior's total strength.

### Phase 1: Ground-Truth Routing

```bash
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --tag coda_phase1 \
    --temporal_priors \
    --augment \
    --ir_path data/irs/openair \
    --cold_start_prob 0.15 \
    --jump_prob 0.10 \
    --loss_calibration uncertainty \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 5e-4 \
    --amp \
    --amp_dtype bfloat16
```

### Phase 2: Scheduled Sampling

Fine-tune from the structured Phase 1 checkpoint so the learned uncertainty
weights are carried into Phase 2:
```bash
python scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --param_path params/PHASE1_RUN/best_checkpoint.pt \
    --tag coda_phase2 \
    --temporal_priors \
    --augment \
    --ir_path data/irs/openair \
    --cold_start_prob 0.15 \
    --jump_prob 0.10 \
    --loss_calibration uncertainty \
    --scheduled_sampling \
    --ss_max_p 0.7 \
    --ss_ramp_epochs 5 \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-4 \
    --amp \
    --amp_dtype bfloat16
```

Resume an interrupted phase in the same output directory with the original
optimizer, scheduler, scaler, loss-weight, and random-number-generator states
by adding `--resume_state params/RUN_NAME/latest_checkpoint.pt` to the same
training command.

Use `--amp_dtype float16` on GPUs without BF16 support. The numerically
sensitive ROI-FiLM convolution automatically runs in FP32 under FP16 autocast.
The smaller cross-attention block runs in FP32 under either AMP dtype to keep
packed Q/K/V backward stable while the convolutional backbone remains mixed
precision.
The training loop also transfers scalar metrics together with the mandatory
finite-gradient norm check, avoiding a second CUDA synchronization on every
optimizer update without weakening the protected-step behavior.

## Evaluation

Evaluation requires a compatible model checkpoint and the `net_config.json`
written beside it during training. Model weights are not included in this
source repository. In the commands below, replace `path/to/best_model.pt`
with the path to either an exported model or a structured training checkpoint.

### Standard Tracking (Single Piece)

```bash
python scripts/evaluate.py \
    --param_path path/to/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName
```

This prints onset error ratios at multiple thresholds, system/bar accuracy, and frame-level pixel error.

### Jump Recovery (Single Piece)

Evaluate on the repeat-aware jump-augmented test set with break mode enabled:
```bash
python scripts/evaluate.py \
    --param_path path/to/best_model.pt \
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
    --param_path path/to/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --label "CODA (ours)" \
    --metrics_dir results/metrics/standard \
    --save_summary results/standard_summary.json

# Jump recovery — Table 2, repeat subset
python scripts/evaluate_batch.py \
    --param_path path/to/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --break_mode \
    --label "CODA (full) - repeat" \
    --metrics_dir results/metrics/repeat \
    --save_summary results/repeat_summary.json

# Jump recovery — Table 2, random subset
python scripts/evaluate_batch.py \
    --param_path path/to/best_model.pt \
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
    --param_path path/to/best_model.pt \
    --test_dir data/msmd/msmd_test \
    --test_piece PieceName \
    --output_dir videos/
```

Use `--plot` to also display the visualization in a live window. Use `--no_video` to skip video generation (metrics only).

To batch-evaluate with inline video generation:
```bash
python scripts/evaluate_batch.py \
    --param_path path/to/best_model.pt \
    --test_dir data/msmd/msmd_test_jump/repeat \
    --break_mode \
    --label "CODA (full) - repeat" \
    --metrics_dir results/metrics/repeat \
    --save_summary results/repeat_summary.json \
    --with_video --video_dir results/videos/repeat
```

Videos are generated inline (one piece at a time) during the metrics pass.

## Citation

```bibtex
@inproceedings{yang2026coda,
  title={{CODA}: Cascaded Online Discontinuity-Aware Alignment for Real-Time Score Following},
  author={Yang, Yining and Chen, Ruogu and Han, Jie},
  booktitle={Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  address={Abu Dhabi, UAE},
  year={2026}
}
```

## Acknowledgments

- Preprocessed [MSMD](https://github.com/CPJKU/msmd) data provided by [CYOLO Score Following](https://github.com/CPJKU/cyolo_score_following) (Henkel & Widmer, 2021)
- Audio encoder based on [Mamba](https://github.com/state-spaces/mamba)
