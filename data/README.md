# Data Preparation

## MSMD Dataset

CODA uses the [Multimodal Sheet Music Dataset (MSMD)](https://zenodo.org/record/4745838).

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
- `coords`: Per-note coordinates `[n_notes, 5]` with `(note_y, note_x, system_idx, bar_idx, page_idx)`
- `bars`: Bar bounding boxes (list of dicts with `x, y, w, h, page_nr`)
- `systems`: System bounding boxes (list of dicts with `x, y, w, h, page_nr`)
- `synthesized`: Boolean flag indicating synthesized audio

### Speed Augmentation

To generate tempo-augmented variants for training:
```bash
# Generate speed-augmented copies at 0.8x and 1.2x tempo
# (Script not included; use your preferred audio time-stretching tool)
```

### Jump Augmentation

Generate jump-augmented variants for jump recovery training/evaluation:
```bash
python scripts/generate_jump_data.py \
    --input_dir data/msmd/msmd_train \
    --output_dir data/msmd/msmd_train_jump \
    --num_variants 3
```
