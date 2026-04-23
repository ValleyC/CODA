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

### Repeat-Aware Jump-Augmented Test Set

Download the [pre-built jump-augmented test set](https://drive.google.com/file/d/12hDvbjYfrRLACsh45mQaOA4tiFcIobTo/view?usp=sharing) directly:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=12hDvbjYfrRLACsh45mQaOA4tiFcIobTo -O data/msmd_test_jump.zip
unzip data/msmd_test_jump.zip -d data/msmd/
```

Alternatively, generate it from the base MSMD test set:
```bash
python scripts/generate_repeat_test.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --annotations data/repeat_annotations.json
```

This produces two subsets:
- `repeat/` — pieces with written repeats, jumps follow annotated performance order
- `random/` — pieces without repeats, random jumps inserted

### Repeat Annotations (`repeat_annotations.json`)

Per-piece annotations for the 94 MSMD test pieces. Each entry contains:
- `has_repeats`: Whether the piece has written repeat structures
- `repeat_type`: Type of repeat (`"binary"`, `"first_half"`, `"da_capo"`, `"ternary"`)
- `performance_order`: List of `[start_bar, end_bar]` segments (inclusive, 0-indexed) in the order a performer would play them

Example:
```json
{
  "BachJS__BWV994__bach-applicatio_synth": {
    "bars": 8,
    "has_repeats": true,
    "repeat_type": "binary",
    "performance_order": [[0, 3], [0, 3], [4, 7], [4, 7]]
  }
}
```
