"""
Generate a repeat-aware jump-augmented test benchmark.

For pieces WITH written repeats (annotated in repeat_annotations.json),
jumps follow the actual repeat structure of the score. For pieces WITHOUT
repeats, random jumps are inserted.

This produces two subsets:
  - repeat/   : pieces with real repeat structures
  - random/   : pieces with random jumps

Usage:
    python scripts/generate_repeat_test.py \
        --input_dir data/msmd/msmd_test \
        --output_dir data/msmd/msmd_test_jump \
        --annotations data/repeat_annotations.json
"""

import os
import json
import random
import shutil
import argparse
import numpy as np
from typing import List, Dict, Tuple

from coda.utils.data_utils import load_piece, FPS, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE
from coda.utils.general import xywh2xyxy

from generate_jump_test_data import (
    build_sequences, build_jump_indices, select_jump_types,
    select_jump_positions, inject_jumps,
)


def build_bar_frame_ranges(n_frames: int, interpol_fnc) -> Dict[int, Tuple[int, int]]:
    """Map each bar_idx to its (start_frame, end_frame) range."""
    bar_frames = {}
    for frame in range(n_frames):
        pos = np.asarray(interpol_fnc(frame), dtype=np.int32)
        bar_idx = int(pos[3])
        if bar_idx not in bar_frames:
            bar_frames[bar_idx] = [frame, frame]
        else:
            bar_frames[bar_idx][1] = frame
    return {k: tuple(v) for k, v in bar_frames.items()}


def build_sequence_entry(frame, interpol_fnc, start_frame, bars, systems,
                         onsets, synthesized, scale_factor, padded_scores,
                         page_systems_cache):
    """Build a single sequence dict entry for a given frame."""
    true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
    bar_idx = int(true_position[3])
    system_idx = int(true_position[2])
    page_nr = int(true_position[-1])

    bar = bars[bar_idx]
    system = systems[system_idx]

    page_systems = page_systems_cache.get(page_nr, [])
    if page_systems:
        systems_xywh = np.asarray([[s['x'], s['y'], s['w'], s['h']] for s in page_systems])
        systems_xyxy = xywh2xyxy(systems_xywh)
        max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                       int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
        max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                       max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))
    else:
        max_x_shift = (0, 0)
        max_y_shift = (0, 0)

    return {
        'piece_id': 0,
        'is_onset': frame in onsets,
        'start_frame': start_frame,
        'frame': frame,
        'true_position': true_position,
        'true_system': np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=float),
        'true_bar': np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=float),
        'height': system['h'],
        'synthesized': synthesized,
        'scale_factor': scale_factor,
        'max_x_shift': max_x_shift,
        'max_y_shift': max_y_shift,
    }


def generate_repeat_sequence(
    piece_path: str,
    piece_name: str,
    performance_order: List[List[int]],
    scale_width: int = 416,
    silence_frames: int = 5,
) -> Tuple[List[dict], List[dict]]:
    """
    Generate a sequence following the annotated repeat structure.

    At each jump point (where performance_order goes backward or to a
    non-adjacent location), silence frames are inserted by resetting the
    audio context window.

    Returns:
        (sequences, jump_metadata)
    """
    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = \
        load_piece(piece_path, piece_name)

    n_frames = int(np.ceil(FPS * signal.shape[0] / SAMPLE_RATE))
    scale_factor = padded_scores.shape[1] / scale_width

    bar_frame_ranges = build_bar_frame_ranges(n_frames, interpol_fnc)

    # Page-system cache
    page_systems_cache = {}
    valid_pages = np.unique(coords_new[:, -1])
    for page_idx in valid_pages:
        page_systems_cache[int(page_idx)] = [s for s in systems if s['page_nr'] == page_idx]

    onsets_set = set(onsets)
    sequences = []
    jump_metadata = []
    prev_end_bar = -1
    prev_seg_end_frame = None

    for seg_idx, (start_bar, end_bar) in enumerate(performance_order):
        # Determine if this is a jump (non-sequential transition)
        is_jump = (seg_idx > 0 and start_bar <= prev_end_bar)

        # Get frame range for this segment
        seg_start_frame = None
        seg_end_frame = None
        for b in range(start_bar, end_bar + 1):
            if b in bar_frame_ranges:
                bf_start, bf_end = bar_frame_ranges[b]
                if seg_start_frame is None:
                    seg_start_frame = bf_start
                seg_end_frame = bf_end

        if seg_start_frame is None:
            print(f"  Warning: bars {start_bar}-{end_bar} not found in frame ranges")
            continue

        # At jump points, insert silence frames before the destination
        if is_jump:
            # Record jump metadata
            src_pos = np.asarray(interpol_fnc(prev_seg_end_frame), dtype=np.int32)
            dest_pos = np.asarray(interpol_fnc(seg_start_frame), dtype=np.int32)
            jump_metadata.append({
                'output_idx': len(sequences),
                'src_frame': prev_seg_end_frame,
                'dest_frame': seg_start_frame,
                'jump_type': 'repeat',
                'silence_frames': silence_frames,
                'post_silence_frames': 0,
                'same_page': int(src_pos[-1]) == int(dest_pos[-1]),
                'src_bar': int(src_pos[3]),
                'dest_bar': int(dest_pos[3]),
                'src_page': int(src_pos[-1]),
                'dest_page': int(dest_pos[-1]),
            })

            for s in range(silence_frames):
                entry = build_sequence_entry(
                    seg_start_frame, interpol_fnc, seg_start_frame, bars, systems,
                    onsets_set, synthesized, scale_factor, padded_scores,
                    page_systems_cache
                )
                entry['is_silence'] = True
                entry['is_post_silence'] = False
                if s == 0:
                    entry['is_jump_destination'] = True
                    entry['silence_frames'] = silence_frames
                sequences.append(entry)

        # Build entries for each frame in this segment
        audio_start = seg_start_frame
        for frame in range(seg_start_frame, seg_end_frame + 1):
            entry = build_sequence_entry(
                frame, interpol_fnc, audio_start, bars, systems,
                onsets_set, synthesized, scale_factor, padded_scores,
                page_systems_cache
            )
            entry['is_silence'] = False
            entry['is_post_silence'] = False
            sequences.append(entry)

        prev_end_bar = end_bar
        prev_seg_end_frame = seg_end_frame

    return sequences, jump_metadata


def generate_random_jump_sequence(
    piece_path: str,
    piece_name: str,
    scale_width: int = 416,
    min_jumps: int = 3,
    min_gap_sec: float = 3.0,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Generate a sequence with diverse random jumps (for pieces without repeats).

    Uses the full pipeline from generate_jump_test_data.py:
    - Diverse jump types (repeat, skip, restart, bar_correction, page_jump, random)
    - Onset-biased destination sampling
    - Random 3-12 silence frames + 1-8 post-silence frames per jump
    - Well-separated jump positions

    Returns:
        (output_sequences, jump_metadata)
    """
    # Build per-frame sequences with gt_valid tracking
    sequences, systems, bars, page_metadata = build_sequences(
        piece_path, piece_name, scale_width
    )
    n_frames = len(sequences)

    if n_frames < 80:
        return [], []

    # Determine piece properties
    all_pages = set(int(seq['true_position'][-1]) for seq in sequences)
    is_multi_page = len(all_pages) > 1

    # Build indices for destination sampling
    system_map, page_map = build_jump_indices(sequences)

    # Select diverse jump types
    jump_types = select_jump_types(min_jumps, is_multi_page)

    # Select well-separated jump positions
    min_gap_frames = int(min_gap_sec * FPS)
    jump_positions = select_jump_positions(sequences, min_jumps, min_gap_frames)

    # Inject jumps with random silence durations
    output_seqs, jumps = inject_jumps(
        sequences, system_map, page_map, jump_types, jump_positions,
        seed=seed
    )

    return output_seqs, jumps


def save_variant(input_dir, piece_name, sequences, output_path, scale_width=416,
                 jump_metadata=None):
    """Save a jump-augmented variant as NPZ."""
    original_npz = os.path.join(input_dir, f'{piece_name}.npz')
    original = np.load(original_npz, allow_pickle=True)

    save_kwargs = dict(
        sheets=original['sheets'],
        coords=original['coords'],
        systems=original['systems'],
        bars=original['bars'],
        synthesized=original['synthesized'],
        sequences=np.array(sequences, dtype=object),
        audio_source=piece_name,
        scale_width=scale_width,
    )
    if jump_metadata is not None:
        save_kwargs['jump_metadata'] = np.array(jump_metadata, dtype=object)

    np.savez(output_path, **save_kwargs)


def copy_wav(input_dir, output_dir, piece_name):
    """Copy WAV file to output directory if not already present."""
    src = os.path.join(input_dir, f'{piece_name}.wav')
    dst = os.path.join(output_dir, f'{piece_name}.wav')
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description='Generate repeat-aware jump-augmented test benchmark'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing original test .npz and .wav files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory (repeat/ and random/ subdirs created)')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to repeat_annotations.json')
    parser.add_argument('--scale_width', type=int, default=416)
    parser.add_argument('--repeat_silence_frames', type=int, default=2,
                        help='Silence frames at repeat jumps (default: 2, ~100ms)')
    parser.add_argument('--min_jumps', type=int, default=3,
                        help='Minimum random jumps per piece (default: 3)')
    parser.add_argument('--min_gap_sec', type=float, default=3.0,
                        help='Minimum gap between random jumps in seconds (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--piece_name', type=str, default=None,
                        help='Process a single piece (for testing)')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.annotations) as f:
        annotations = json.load(f)

    repeat_dir = os.path.join(args.output_dir, 'repeat')
    random_dir = os.path.join(args.output_dir, 'random')
    os.makedirs(repeat_dir, exist_ok=True)
    os.makedirs(random_dir, exist_ok=True)

    # Get piece list
    if args.piece_name:
        piece_names = [args.piece_name]
    else:
        npz_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npz')])
        piece_names = [f[:-4] for f in npz_files]

    n_repeat = 0
    n_random = 0
    n_skipped = 0

    for piece_name in piece_names:
        ann = annotations.get(piece_name)
        if ann is None:
            print(f"  Warning: {piece_name} not in annotations, skipping")
            n_skipped += 1
            continue

        has_repeats = ann.get('has_repeats', False)
        perf_order = ann.get('performance_order')

        if has_repeats and perf_order:
            # Repeat subset: follow annotated structure
            print(f"  [repeat] {piece_name}")
            try:
                sequences, jump_metadata = generate_repeat_sequence(
                    args.input_dir, piece_name, perf_order,
                    args.scale_width, args.repeat_silence_frames,
                )
                if sequences:
                    out_path = os.path.join(repeat_dir, f'{piece_name}.npz')
                    save_variant(args.input_dir, piece_name, sequences, out_path,
                                 args.scale_width, jump_metadata=jump_metadata)
                    copy_wav(args.input_dir, repeat_dir, piece_name)
                    n_repeat += 1
                    for j in jump_metadata:
                        print(f"    [repeat] bar {j['src_bar']}->{j['dest_bar']} "
                              f"silence={j['silence_frames']}f")
                else:
                    print(f"    Empty sequence, skipping")
                    n_skipped += 1
            except Exception as e:
                print(f"    Error: {e}")
                n_skipped += 1
        else:
            # Random subset: insert diverse random jumps
            print(f"  [random] {piece_name}")
            try:
                sequences, jump_metadata = generate_random_jump_sequence(
                    args.input_dir, piece_name,
                    args.scale_width, args.min_jumps, args.min_gap_sec,
                    seed=args.seed,
                )
                if sequences:
                    out_path = os.path.join(random_dir, f'{piece_name}.npz')
                    save_variant(args.input_dir, piece_name, sequences, out_path,
                                 args.scale_width, jump_metadata=jump_metadata)
                    copy_wav(args.input_dir, random_dir, piece_name)
                    n_random += 1
                    # Print jump summary
                    for j in jump_metadata:
                        same = "same-page" if j['same_page'] else f"page {j['src_page']}->{j['dest_page']}"
                        print(f"    [{j['jump_type']}] sys {j['src_sys']}->{j['dest_sys']} "
                              f"({same}) silence={j['silence_frames']}f post={j['post_silence_frames']}f")
                else:
                    print(f"    Empty sequence (too short), skipping")
                    n_skipped += 1
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                n_skipped += 1

    print(f"\nDone.")
    print(f"  Repeat subset: {n_repeat} pieces -> {repeat_dir}")
    print(f"  Random subset: {n_random} pieces -> {random_dir}")
    print(f"  Skipped: {n_skipped}")

    # Save manifest
    manifest = {
        'repeat_pieces': n_repeat,
        'random_pieces': n_random,
        'repeat_silence_frames': args.repeat_silence_frames,
        'random_min_jumps': args.min_jumps,
        'random_min_gap_sec': args.min_gap_sec,
        'seed': args.seed,
    }
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
