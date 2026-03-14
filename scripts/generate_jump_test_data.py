"""
Generate jump-augmented test data matching the current training pipeline.

Takes a regular piece NPZ and injects jumps with the same silence/post-silence
structure used during training:
  - JUMP_SILENCE_MIN/MAX_FRAMES (3-12) silence frames (audio zeroed)
  - JUMP_POST_SILENCE_MIN/MAX_FRAMES (1-8) post-silence frames at destination
  - Onset-biased destination sampling

Testing policy (deterministic, balanced, interpretable):
  - Each piece gets at least --min_jumps (default 3) jumps
  - Jump types are diverse: distinct types when feasible
  - page_jump is excluded on single-page pieces
  - Jumps are well separated in time (--min_gap_sec, default 3s)
  - If a sampled type is infeasible, resample rather than silent fallback

Silence frames carry *destination* GT (matching training semantics in dataset.py
where seq is swapped to destination before silence insertion).

Jump evaluation uses saved jump_metadata (not frame discontinuity heuristics)
to avoid the multi-detection problem with frame=-1 markers.

Usage:
    python scripts/generate_jump_test_data.py \
        --test_dir data/msmd/msmd_test \
        --test_piece BachJS__BWV269__bwv_269_synth \
        --output_dir data/msmd/msmd_jump_test \
        --min_jumps 3 \
        --scale_width 416
"""

import argparse
import os
import random
import shutil
import numpy as np

from coda.utils.data_utils import (
    load_piece, SAMPLE_RATE, HOP_SIZE, FPS, FRAME_SIZE
)
from coda.utils.general import xywh2xyxy
from coda.dataset import (
    JUMP_TYPE_WEIGHTS,
    JUMP_SILENCE_MIN_FRAMES, JUMP_SILENCE_MAX_FRAMES,
    JUMP_POST_SILENCE_MIN_FRAMES, JUMP_POST_SILENCE_MAX_FRAMES,
)


def build_sequences(path, piece_name, scale_width):
    """Build per-frame sequences from a regular piece, mirroring data_utils.py logic.

    Uses the same gt_valid tracking as the main data pipeline.
    """
    padded_scores, org_scores, onsets, coords, bars, systems, interpol_fnc, signal, pad, _ = \
        load_piece(path, piece_name)

    n_pages = padded_scores.shape[0]
    h = padded_scores.shape[1]
    scale_factor = h / scale_width

    # Match data_utils.py frame count (ceil, not floor)
    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))

    # Build page metadata — same as data_utils.py
    page_systems_dict = {}
    for s in systems:
        pg = s['page_nr']
        if pg not in page_systems_dict:
            page_systems_dict[pg] = []
        page_systems_dict[pg].append(s)

    valid_pages = sorted(set(s['page_nr'] for s in systems))
    system_global_to_local = {}
    bar_global_to_local = {}
    page_metadata = {}

    for pg in valid_pages:
        p_sys = [s for s in systems if s['page_nr'] == pg]
        p_bar = [b for b in bars if b['page_nr'] == pg]

        # Global -> local index maps
        sys_global_indices = [i for i, s in enumerate(systems) if s['page_nr'] == pg]
        for local_i, global_i in enumerate(sys_global_indices):
            system_global_to_local[global_i] = local_i

        bar_global_indices = [i for i, b in enumerate(bars) if b['page_nr'] == pg]
        for local_i, global_i in enumerate(bar_global_indices):
            bar_global_to_local[global_i] = local_i

        sys_boxes = np.array([[s['x'], s['y'], s['w'], s['h']] for s in p_sys],
                             dtype=np.float32) if p_sys else np.zeros((0, 4), dtype=np.float32)
        bar_boxes = np.array([[b['x'], b['y'], b['w'], b['h']] for b in p_bar],
                             dtype=np.float32) if p_bar else np.zeros((0, 4), dtype=np.float32)

        # bars_per_system via spatial containment
        bars_per_system = []
        for sys_dict in p_sys:
            sx1 = sys_dict['x'] - sys_dict['w'] / 2
            sy1 = sys_dict['y'] - sys_dict['h'] / 2
            sx2 = sys_dict['x'] + sys_dict['w'] / 2
            sy2 = sys_dict['y'] + sys_dict['h'] / 2
            member_bars = [bi for bi, bd in enumerate(p_bar)
                           if sx1 <= bd['x'] <= sx2 and sy1 <= bd['y'] <= sy2]
            bars_per_system.append(member_bars)

        page_metadata[pg] = {
            'system_boxes': sys_boxes,
            'bar_boxes': bar_boxes,
            'bars_per_system': bars_per_system,
        }

    # Build frame-by-frame sequences with gt_valid tracking
    sequences = []
    curr_page = -1
    start_frame = 0
    n_invalid = 0
    prev_sys_for_tp = -1
    prev_bar_page_for_tp = -1

    for frame in range(n_frames):
        true_position = np.asarray(interpol_fnc(frame), dtype=np.int32).flatten()
        bar_idx = int(true_position[3])
        system_idx = int(true_position[2])
        page_nr = int(true_position[-1])

        if page_nr != curr_page:
            curr_page = page_nr
            start_frame = frame
            prev_sys_for_tp = -1
            prev_bar_page_for_tp = -1

        bar = bars[bar_idx]
        system = systems[system_idx]

        true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=float)
        true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=float)

        # Validate GT mappings — same as data_utils.py line 353-369
        gt_sys_local = system_global_to_local.get(system_idx, None)
        gt_bar_local = bar_global_to_local.get(bar_idx, None)
        gt_valid = True

        if gt_sys_local is None:
            gt_valid = False
            gt_sys_local = 0
            gt_bar_in_sys = 0
        else:
            bps = page_metadata.get(curr_page, {}).get('bars_per_system', [])
            bars_in_sys = bps[gt_sys_local] if gt_sys_local < len(bps) else []
            if gt_bar_local is None or gt_bar_local not in bars_in_sys:
                gt_valid = False
                gt_bar_in_sys = 0
            else:
                gt_bar_in_sys = bars_in_sys.index(gt_bar_local)

        if not gt_valid:
            n_invalid += 1

        sequences.append({
            'piece_id': 0,
            'is_onset': frame in onsets,
            'start_frame': start_frame,
            'frame': frame,
            'true_position': true_position,
            'true_system': true_system,
            'true_bar': true_bar,
            'height': system['h'],
            'synthesized': True,
            'scale_factor': scale_factor,
            'gt_system_page_idx': gt_sys_local,
            'gt_bar_in_system_idx': gt_bar_in_sys,
            'gt_valid': gt_valid,
            'prev_gt_system_page_idx': prev_sys_for_tp,
            'prev_gt_bar_page_idx': prev_bar_page_for_tp,
        })

        if gt_valid:
            prev_sys_for_tp = gt_sys_local
            prev_bar_page_for_tp = gt_bar_local

    if n_invalid > 0:
        print(f"  [build_sequences] {n_invalid}/{n_frames} frames have invalid GT "
              f"({n_invalid/n_frames*100:.1f}%)")

    return sequences, systems, bars, page_metadata


def build_jump_indices(sequences):
    """Build system_map and page_map for jump destination sampling.

    Only includes gt_valid frames to avoid generating jumps to invalid targets.
    """
    system_map = {}  # (page, sys_page_idx) -> [seq indices]
    page_map = {}    # page -> [seq indices]

    for i, seq in enumerate(sequences):
        if not seq.get('gt_valid', True):
            continue
        page = int(seq['true_position'][-1])
        sys_idx = seq['gt_system_page_idx']
        key = (page, sys_idx)
        if key not in system_map:
            system_map[key] = []
        system_map[key].append(i)
        if page not in page_map:
            page_map[page] = []
        page_map[page].append(i)

    return system_map, page_map


def find_jump_destination(sequences, src_idx, system_map, page_map, jump_type):
    """Find a destination for a specific jump type. Returns (dest_idx, actual_type) or (None, type)."""
    src_seq = sequences[src_idx]
    src_page = int(src_seq['true_position'][-1])
    src_sys = src_seq['gt_system_page_idx']
    src_bar_in_sys = src_seq['gt_bar_in_system_idx']

    dest_indices = None

    if jump_type == 'repeat':
        if random.random() < 0.7:
            offsets = [1, 2]
        else:
            offsets = [3, 4]
        random.shuffle(offsets)
        for offset in offsets:
            target_sys = src_sys - offset
            if target_sys >= 0 and (src_page, target_sys) in system_map:
                dest_indices = system_map[(src_page, target_sys)]
                break

    elif jump_type == 'skip':
        offsets = [1, 2, 3]
        random.shuffle(offsets)
        for offset in offsets:
            target_sys = src_sys + offset
            if (src_page, target_sys) in system_map:
                dest_indices = system_map[(src_page, target_sys)]
                break

    elif jump_type == 'restart':
        if src_sys != 0 and (src_page, 0) in system_map:
            dest_indices = system_map[(src_page, 0)]

    elif jump_type == 'page_jump':
        other_pages = [p for p in page_map if p != src_page]
        if other_pages:
            dest_page = random.choice(other_pages)
            dest_indices = page_map[dest_page]

    elif jump_type == 'bar_correction':
        same_sys = system_map.get((src_page, src_sys), [])
        diff_bar = [i for i in same_sys
                    if sequences[i]['gt_bar_in_system_idx'] != src_bar_in_sys]
        if diff_bar:
            dest_indices = diff_bar
        else:
            for offset in [1, -1]:
                adj = (src_page, src_sys + offset)
                if adj in system_map:
                    dest_indices = system_map[adj]
                    break

    elif jump_type == 'random':
        all_indices = []
        for key, indices in system_map.items():
            if key != (src_page, src_sys):
                all_indices.extend(indices)
        if all_indices:
            dest_indices = all_indices

    if dest_indices is None:
        return None, jump_type

    # Onset bias: prefer onset frames
    onset_indices = [i for i in dest_indices if sequences[i].get('is_onset', False)]
    if onset_indices:
        dest_idx = random.choice(onset_indices)
    else:
        dest_idx = random.choice(dest_indices)

    return dest_idx, jump_type


def select_jump_types(n_jumps, is_multi_page):
    """Select diverse jump types for a test piece.

    Rules:
    - At least 3 distinct types when feasible
    - Exclude page_jump on single-page pieces
    - At most 1 page_jump on multi-page pieces (unless stress-testing)
    """
    available = ['repeat', 'skip', 'restart', 'bar_correction', 'random']
    if is_multi_page:
        available.append('page_jump')

    # Pick distinct types first
    n_distinct = min(n_jumps, len(available))
    types = random.sample(available, n_distinct)

    # If we need more jumps than distinct types, add more from available
    while len(types) < n_jumps:
        # Don't add extra page_jumps
        extra_pool = [t for t in available if t != 'page_jump']
        types.append(random.choice(extra_pool))

    # Enforce at most 1 page_jump
    page_jump_count = types.count('page_jump')
    if page_jump_count > 1:
        for i in range(len(types)):
            if types[i] == 'page_jump' and page_jump_count > 1:
                types[i] = random.choice([t for t in available if t != 'page_jump'])
                page_jump_count -= 1

    return types


def select_jump_positions(sequences, n_jumps, min_gap_frames, margin_frames=40):
    """Select well-separated frame positions for jumps from valid-source frames only.

    Ensures:
    - Positions are gt_valid frames (matching training: jumps only from valid samples)
    - At least margin_frames from start/end
    - At least min_gap_frames between consecutive jumps
    - All positions are within sequence bounds
    """
    n_frames = len(sequences)

    # Collect valid frame indices
    valid_indices = [i for i, seq in enumerate(sequences)
                     if seq.get('gt_valid', True)
                     and margin_frames <= i < n_frames - margin_frames]

    if not valid_indices:
        # Fallback: use all frames with margin
        valid_indices = list(range(margin_frames, max(margin_frames + 1, n_frames - margin_frames)))

    if not valid_indices:
        # Piece extremely short, use whatever we have
        valid_indices = list(range(n_frames))

    usable_start = valid_indices[0]
    usable_end = valid_indices[-1]
    usable_range = usable_end - usable_start

    if usable_range <= 0:
        # Single valid frame, replicate
        return [valid_indices[0]] * n_jumps

    # Space jumps evenly across valid range, then snap to nearest valid frame
    positions = []
    segment_size = usable_range / (n_jumps + 1)

    for i in range(n_jumps):
        center = usable_start + int(segment_size * (i + 1))
        # Add jitter up to 20% of segment_size
        max_jitter = min(int(segment_size * 0.2), min_gap_frames // 2)
        jitter = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
        target = center + jitter

        # Snap to nearest valid frame
        best = min(valid_indices, key=lambda x: abs(x - target))
        positions.append(best)

    # Enforce minimum gap: push later positions forward, snapping to valid frames
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap_frames:
            target = positions[i - 1] + min_gap_frames
            # Find nearest valid frame >= target
            candidates = [v for v in valid_indices if v >= target]
            if candidates:
                positions[i] = candidates[0]
            else:
                # No valid frame far enough ahead; use the last valid frame
                positions[i] = valid_indices[-1]

    # Clamp all positions to valid range
    positions = [max(0, min(n_frames - 1, p)) for p in positions]

    return positions


def inject_jumps(sequences, system_map, page_map, jump_types, jump_positions, seed=42):
    """Inject jumps at specified positions with specified types.

    Silence frames carry *destination* GT (matching training semantics).
    Frame field is set to -1 for silence frames (audio should be zeroed).

    Returns (output_sequences, jump_metadata).
    """
    random.seed(seed)

    # Sort by position
    jump_plan = sorted(zip(jump_positions, jump_types), key=lambda x: x[0])

    output = []
    jumps = []
    jump_idx = 0  # next planned jump
    i = 0

    while i < len(sequences):
        # Check if current frame is a planned jump position
        if jump_idx < len(jump_plan) and i >= jump_plan[jump_idx][0]:
            target_pos, target_type = jump_plan[jump_idx]
            jump_idx += 1

            # Find destination for this type
            dest_idx, actual_type = find_jump_destination(
                sequences, i, system_map, page_map, target_type
            )

            # If infeasible, try other types (resample, don't silently fallback)
            if dest_idx is None:
                alternative_types = [t for t in JUMP_TYPE_WEIGHTS.keys()
                                     if t != target_type]
                random.shuffle(alternative_types)
                for alt_type in alternative_types:
                    dest_idx, actual_type = find_jump_destination(
                        sequences, i, system_map, page_map, alt_type
                    )
                    if dest_idx is not None:
                        actual_type = f"{alt_type}(resampled from {target_type})"
                        break

            if dest_idx is None:
                # Try nearby valid frames (±20) before giving up
                found = False
                for offset in range(1, 21):
                    for probe in [i + offset, i - offset]:
                        if 0 <= probe < len(sequences) and sequences[probe].get('gt_valid', True):
                            for try_type in list(JUMP_TYPE_WEIGHTS.keys()):
                                dest_idx, actual_type = find_jump_destination(
                                    sequences, probe, system_map, page_map, try_type
                                )
                                if dest_idx is not None:
                                    actual_type = f"{try_type}(nearby probe from {target_type})"
                                    i = probe  # shift source to this valid frame
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break

            if dest_idx is None:
                # Truly infeasible, skip
                print(f"  WARNING: Could not find any jump destination near frame {i}, skipping")
                normal_entry = dict(sequences[i])
                normal_entry['is_silence'] = False
                normal_entry['is_post_silence'] = False
                output.append(normal_entry)
                i += 1
                continue

            src_seq = sequences[i]
            dest_seq = sequences[dest_idx]
            src_page = int(src_seq['true_position'][-1])
            dest_page = int(dest_seq['true_position'][-1])
            same_page = (src_page == dest_page)

            silence_frames = random.randint(
                JUMP_SILENCE_MIN_FRAMES, JUMP_SILENCE_MAX_FRAMES
            )
            post_silence_frames = random.randint(
                JUMP_POST_SILENCE_MIN_FRAMES, JUMP_POST_SILENCE_MAX_FRAMES
            )

            jump_info = {
                'output_idx': len(output),
                'src_frame': i,
                'dest_frame': dest_idx,
                'jump_type': actual_type,
                'silence_frames': silence_frames,
                'post_silence_frames': post_silence_frames,
                'same_page': same_page,
                'src_sys': src_seq['gt_system_page_idx'],
                'dest_sys': dest_seq['gt_system_page_idx'],
                'src_page': src_page,
                'dest_page': dest_page,
            }
            jumps.append(jump_info)

            # Insert silence frames — DESTINATION GT (matching training)
            for sf in range(silence_frames):
                silence_entry = dict(dest_seq)  # destination GT, not source
                silence_entry['frame'] = -1     # marker for zero audio
                silence_entry['is_silence'] = True
                silence_entry['is_post_silence'] = False
                output.append(silence_entry)

            # Insert post-silence frames (destination audio + destination GT)
            for pf in range(post_silence_frames):
                post_idx = min(dest_idx + pf, len(sequences) - 1)
                post_entry = dict(sequences[post_idx])
                post_entry['is_silence'] = False
                post_entry['is_post_silence'] = True
                output.append(post_entry)

            # Continue from destination (after post-silence)
            i = dest_idx + post_silence_frames
            continue

        # Normal frame
        normal_entry = dict(sequences[i])
        normal_entry['is_silence'] = False
        normal_entry['is_post_silence'] = False
        output.append(normal_entry)
        i += 1

    return output, jumps


def main():
    parser = argparse.ArgumentParser(
        description='Generate jump-augmented test data matching training pipeline')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory with source piece NPZs (e.g. msmd_valid)')
    parser.add_argument('--test_piece', type=str, required=True,
                        help='Piece name (without .npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for jump-augmented NPZs')
    parser.add_argument('--min_jumps', type=int, default=3,
                        help='Minimum jumps per piece (default: 3)')
    parser.add_argument('--scale_width', type=int, default=416,
                        help='Scale width for score images (default: 416)')
    parser.add_argument('--min_gap_sec', type=float, default=3.0,
                        help='Minimum gap between jumps in seconds (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading piece: {args.test_piece}")
    print(f"  Source dir: {args.test_dir}")
    print(f"  Min jumps: {args.min_jumps}")
    print(f"  Min gap: {args.min_gap_sec}s ({int(args.min_gap_sec * FPS)} frames)")
    print(f"  Silence frames: {JUMP_SILENCE_MIN_FRAMES}-{JUMP_SILENCE_MAX_FRAMES}")
    print(f"  Post-silence frames: {JUMP_POST_SILENCE_MIN_FRAMES}-{JUMP_POST_SILENCE_MAX_FRAMES}")

    # Build sequences
    sequences, systems, bars, page_metadata = build_sequences(
        args.test_dir, args.test_piece, args.scale_width
    )
    n_frames = len(sequences)
    print(f"  Built {n_frames} frames ({n_frames / FPS:.1f}s)")

    # Determine piece properties
    all_pages = set(int(seq['true_position'][-1]) for seq in sequences)
    is_multi_page = len(all_pages) > 1
    print(f"  Pages: {len(all_pages)}, multi-page: {is_multi_page}")

    # Build indices for destination sampling (only gt_valid frames)
    system_map, page_map = build_jump_indices(sequences)

    # Select jump types (diverse, deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)

    jump_types = select_jump_types(args.min_jumps, is_multi_page)
    print(f"  Jump types: {jump_types}")

    # Select jump positions (well-separated, valid-source only)
    min_gap_frames = int(args.min_gap_sec * FPS)
    jump_positions = select_jump_positions(sequences, args.min_jumps, min_gap_frames)
    print(f"  Jump positions (frames): {jump_positions}")

    # Inject jumps
    output_seqs, jumps = inject_jumps(
        sequences, system_map, page_map, jump_types, jump_positions,
        seed=args.seed
    )
    print(f"  Output: {len(output_seqs)} frames ({len(jumps)} jumps injected)")

    # Verify min_jumps guarantee
    if len(jumps) < args.min_jumps:
        print(f"  WARNING: Only {len(jumps)}/{args.min_jumps} jumps were successfully "
              f"injected. Piece may be too short or have limited system diversity.")

    # Print jump summary
    for j in jumps:
        same = "same-page" if j['same_page'] else f"page {j['src_page']}->{j['dest_page']}"
        print(f"    [{j['jump_type']}] @ frame {j['src_frame']}: "
              f"sys {j['src_sys']}->{j['dest_sys']} ({same}) "
              f"silence={j['silence_frames']}f post={j['post_silence_frames']}f")

    # Load original NPZ to copy sheets/coords/systems/bars
    src_npz = np.load(os.path.join(args.test_dir, args.test_piece + '.npz'),
                      allow_pickle=True)

    # Save output NPZ
    output_name = f"{args.test_piece}_jump_v2"
    output_path = os.path.join(args.output_dir, output_name + '.npz')

    np.savez(output_path,
             sheets=src_npz['sheets'],
             coords=src_npz['coords'],
             systems=src_npz['systems'],
             bars=src_npz['bars'],
             synthesized=src_npz['synthesized'],
             sequences=np.array(output_seqs, dtype=object),
             audio_source=args.test_piece,
             scale_width=args.scale_width,
             jump_metadata=np.array(jumps, dtype=object),
             )

    src_npz.close()

    # Copy the wav file if not already in output dir
    src_wav = os.path.join(args.test_dir, args.test_piece + '.wav')
    dst_wav = os.path.join(args.output_dir, args.test_piece + '.wav')
    if not os.path.exists(dst_wav) and os.path.exists(src_wav):
        shutil.copy2(src_wav, dst_wav)
        print(f"  Copied audio to: {dst_wav}")

    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
