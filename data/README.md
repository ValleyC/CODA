# Data Preparation

## MSMD Dataset

CODA uses the preprocessed version of the [Multimodal Sheet Music Dataset (MSMD)](https://github.com/CPJKU/msmd) provided by [Henkel & Widmer (2021)](https://github.com/CPJKU/cyolo_score_following). Each piece is stored as a `.npz` file (score images and annotations) paired with a `.wav` file (synthesized audio at 22050 Hz).

### Download

```bash
wget https://zenodo.org/record/4745838/files/msmd.zip
unzip msmd.zip
```

### Expected Structure

```
data/
  msmd/
    msmd_train/          # 354 training pieces
      PieceName.npz      # Score images, annotations, metadata
      PieceName.wav      # Synthesized audio (22050 Hz)
      ...
    msmd_valid/          # 19 validation pieces
      ...
    msmd_test/           # 94 test pieces
      ...
```

### NPZ File Contents

Each `.npz` file contains:
- `sheets`: Score page images as uint8 arrays `[n_pages, height, width]`
- `coords`: Per-note coordinates (list of dicts with `note_y, note_x, onset, system_idx, bar_idx, page_idx`)
- `bars`: Bar bounding boxes (list of dicts with `x, y, w, h, page_nr`)
- `systems`: System bounding boxes (list of dicts with `x, y, w, h, page_nr`)
- `synthesized`: Boolean flag indicating synthesized audio

### Jump-Augmented Test Set

Generate jump-augmented variants for jump recovery evaluation:
```bash
python scripts/generate_jump_data.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --num_variants 3
```
