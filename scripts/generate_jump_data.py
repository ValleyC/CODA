"""
Jump augmentation for score following training.

Generates jump-augmented sequence variants for pieces, where the performer
"jumps" to different locations (repeats, skips, restarts) during playback.

This teaches the model to handle non-monotonic progressions and enables
system-level relocalization.

Usage:
    # Generate jump variants for all pieces in a directory
    python scripts/generate_jump_data.py \
        --input_dir path/to/train_data \
        --output_dir path/to/jump_augmented_data \
        --num_variants 3

    # Generate for a single piece
    python scripts/generate_jump_data.py \
        --input_dir path/to/train_data \
        --output_dir path/to/jump_augmented_data \
        --piece_name IMSLP00123 \
        --num_variants 5
"""

import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional

from coda.utils.data_utils import load_piece, FPS, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE
from coda.utils.general import xywh2xyxy


# Jump type distribution (must sum to 1.0)
JUMP_PATTERNS = {
    'repeat': 0.40,      # Jump back 1-4 systems
    'skip': 0.15,        # Jump forward 2-5 systems
    'random': 0.20,      # Jump to arbitrary system
    'restart': 0.15,     # Jump to beginning of piece
    'page_jump': 0.10,   # Jump to previous/next page
}


def build_system_frame_ranges(coords_new: np.ndarray, n_frames: int,
                               interpol_fnc) -> Dict[int, Tuple[int, int]]:
    """
    Build a mapping from system_idx to (start_frame, end_frame).

    Args:
        coords_new: Coordinate array from load_piece
        n_frames: Total number of frames in the piece
        interpol_fnc: Interpolation function from load_piece

    Returns:
        Dict mapping system_idx -> (start_frame, end_frame)
    """
    system_frames = {}

    for frame in range(n_frames):
        true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
        system_idx = int(true_position[2])

        if system_idx not in system_frames:
            system_frames[system_idx] = [frame, frame]
        else:
            system_frames[system_idx][1] = frame

    # Convert to tuples
    return {k: tuple(v) for k, v in system_frames.items()}


def get_system_at_frame(frame: int, interpol_fnc) -> int:
    """Get the system index at a given frame."""
    true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
    return int(true_position[2])


def select_jump_destination(current_system: int, n_systems: int,
                            jump_type: str = None) -> int:
    """
    Select a destination system for a jump.

    Args:
        current_system: Current system index
        n_systems: Total number of systems
        jump_type: Type of jump (repeat, skip, random, restart, page_jump)

    Returns:
        Destination system index
    """
    if jump_type is None:
        # Sample jump type based on distribution
        r = random.random()
        cumsum = 0
        for jt, prob in JUMP_PATTERNS.items():
            cumsum += prob
            if r < cumsum:
                jump_type = jt
                break

    if jump_type == 'repeat':
        # Jump back 1-4 systems
        if current_system == 0:
            # Can't go back from start, fall back to random jump
            candidates = [i for i in range(n_systems) if i != current_system]
            dest = random.choice(candidates) if candidates else current_system
        else:
            offset = random.randint(1, min(4, current_system))
            dest = max(0, current_system - offset)

    elif jump_type == 'skip':
        # Jump forward 2-5 systems
        max_offset = max(2, n_systems - current_system - 1)
        offset = random.randint(2, max(2, min(5, max_offset)))
        dest = min(n_systems - 1, current_system + offset)

    elif jump_type == 'restart':
        # Jump to beginning
        dest = 0

    elif jump_type == 'page_jump':
        # Jump to a random system on a different "page" (approximated as far systems)
        if n_systems <= 2:
            # Not enough systems for page jump, fall back to random
            candidates = [i for i in range(n_systems) if i != current_system]
            dest = random.choice(candidates) if candidates else current_system
        elif current_system < n_systems // 2:
            dest = random.randint(n_systems // 2, n_systems - 1)
        else:
            dest = random.randint(0, max(0, n_systems // 2 - 1))

    else:  # random
        # Jump to any system (different from current)
        candidates = [i for i in range(n_systems) if i != current_system]
        dest = random.choice(candidates) if candidates else current_system

    return dest


def plan_jump_locations(n_frames: int, num_jumps: int,
                        min_gap: int = 100) -> List[int]:
    """
    Plan evenly-spaced jump locations with some randomness.

    Args:
        n_frames: Total number of frames
        num_jumps: Number of jumps to plan
        min_gap: Minimum frames between jumps

    Returns:
        List of frame indices where jumps should occur
    """
    if num_jumps <= 0 or n_frames < min_gap * (num_jumps + 1):
        return []

    # Calculate ideal spacing
    segment_length = n_frames // (num_jumps + 1)

    jump_frames = []
    for i in range(1, num_jumps + 1):
        # Base position
        base = i * segment_length
        # Add randomness (up to 20% of segment length)
        jitter = random.randint(-segment_length // 5, segment_length // 5)
        frame = max(min_gap, min(n_frames - min_gap, base + jitter))
        jump_frames.append(frame)

    return sorted(jump_frames)


def generate_jump_sequences(
    piece_path: str,
    piece_name: str,
    scale_width: int = 416,
    num_variants: int = 3,
    min_play_between_jumps: int = 100,  # 5 sec @ 20 FPS
    min_play_after_jump: int = 40,      # 2 sec @ 20 FPS
) -> List[List[dict]]:
    """
    Generate jump-augmented sequence variants for a piece.

    Key insight: sequences are defined by 'frame' number.
    Labels (true_position, true_system, true_bar) are derived from frame
    via interpol_fnc(frame). So jumping means:
    1. Change frame sequence (e.g., 0,1,2,...,100, JUMP, 50,51,52,...)
    2. Reset start_frame after jump (fresh audio context)
    3. Labels automatically follow because they're computed from frame

    Args:
        piece_path: Directory containing the piece
        piece_name: Name of the piece (without .npz extension)
        scale_width: Width to scale score images to
        num_variants: Number of jump variants to generate
        min_play_between_jumps: Minimum frames between jumps
        min_play_after_jump: Minimum frames to play after a jump

    Returns:
        List of sequence lists (one per variant)
    """
    # Load piece data (same as load_sequences does)
    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = \
        load_piece(piece_path, piece_name)

    n_frames = int(np.ceil(FPS * signal.shape[0] / SAMPLE_RATE))
    scale_factor = padded_scores.shape[1] / scale_width

    # Build system frame ranges for jump targeting
    system_frame_ranges = build_system_frame_ranges(coords_new, n_frames, interpol_fnc)
    n_systems = len(system_frame_ranges)

    if n_systems < 2:
        print(f"Warning: {piece_name} has only {n_systems} system(s), skipping jump augmentation")
        return []

    # Determine jumps per variant based on duration
    duration_sec = n_frames / FPS
    if duration_sec < 120:
        jumps_per_variant = random.randint(1, 2)
    elif duration_sec < 300:
        jumps_per_variant = random.randint(2, 4)
    else:
        jumps_per_variant = random.randint(4, 6)

    # Precompute page-specific systems for max shift calculation
    page_systems_cache = {}
    valid_pages = np.unique(coords_new[:, -1])
    for page_idx in valid_pages:
        page_systems_cache[int(page_idx)] = [s for s in systems if s['page_nr'] == page_idx]

    all_variants = []

    for variant_idx in range(num_variants):
        # Plan jump points (evenly spaced with randomness)
        jump_points = plan_jump_locations(n_frames, jumps_per_variant, min_play_between_jumps)

        if not jump_points:
            continue

        # Build sequence with jumps
        sequences = []
        start_frame = 0    # Audio context start

        jump_idx = 0
        frame = 0
        next_allowed_jump = 0  # Track when next jump is allowed

        while frame < n_frames:
            # Check if we should jump (only if past cooldown)
            if (jump_idx < len(jump_points) and
                frame >= jump_points[jump_idx] and
                frame >= next_allowed_jump):

                # Select jump destination
                current_system = get_system_at_frame(frame, interpol_fnc)
                dest_system = select_jump_destination(current_system, n_systems)

                if dest_system in system_frame_ranges:
                    dest_frame = system_frame_ranges[dest_system][0]  # Start of destination system

                    # Reset audio context
                    start_frame = dest_frame
                    frame = dest_frame

                    # Set cooldown: no jumps for min_play_after_jump frames
                    next_allowed_jump = frame + min_play_after_jump

                jump_idx += 1

            # Build sequence entry for this frame
            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)
            bar_idx = int(true_position[3])
            system_idx = int(true_position[2])
            page_nr = int(true_position[-1])

            # Reset start_frame on page change (existing behavior)
            if sequences and int(sequences[-1]['true_position'][-1]) != page_nr:
                start_frame = frame

            # Get bar and system info
            bar = bars[bar_idx]
            system = systems[system_idx]

            # Compute max shifts (same logic as original load_sequences)
            page_systems = page_systems_cache.get(page_nr, [])
            if page_systems:
                systems_xywh = np.asarray([[x['x'], x['y'], x['w'], x['h']] for x in page_systems])
                systems_xyxy = xywh2xyxy(systems_xywh)

                max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                               int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
                max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                               max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))
            else:
                max_x_shift = (0, 0)
                max_y_shift = (0, 0)

            sequences.append({
                'piece_id': 0,  # Will be assigned during loading
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
            })

            frame += 1

        if sequences:
            all_variants.append(sequences)

    return all_variants


def save_jump_variant(original_npz_path: str, piece_name: str, sequences: List[dict],
                      output_path: str, scale_width: int = 416):
    """
    Save jump-augmented variant.

    Stores sequences array in .npz so loader can detect and use it.
    Also stores audio_source to point to original wav file.

    Args:
        original_npz_path: Path to original .npz file
        piece_name: Original piece name (for audio_source)
        sequences: List of sequence dicts
        output_path: Path to save the new .npz file
        scale_width: Scale width used when generating sequences
    """
    # Load original npz
    original = np.load(original_npz_path, allow_pickle=True)

    # Save with added 'sequences' array, audio_source, and scale_width
    np.savez(
        output_path,
        sheets=original['sheets'],
        coords=original['coords'],
        systems=original['systems'],
        bars=original['bars'],
        synthesized=original['synthesized'],
        sequences=np.array(sequences, dtype=object),  # Pre-computed sequences
        audio_source=piece_name,  # Points to original wav (piece_name.wav)
        scale_width=scale_width,  # For scale_factor validation at load time
    )


def generate_and_save_jump_variants(
    input_dir: str,
    output_dir: str,
    piece_name: str,
    scale_width: int = 416,
    num_variants: int = 3,
    create_symlinks: bool = True,
) -> List[str]:
    """
    Generate jump variants for a piece and save them.

    Args:
        input_dir: Directory containing original .npz and .wav files
        output_dir: Directory to save jump-augmented files
        piece_name: Name of the piece (without extension)
        scale_width: Width to scale score images to
        num_variants: Number of variants to generate
        create_symlinks: Whether to create symlinks for wav files

    Returns:
        List of paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)

    original_npz_path = os.path.join(input_dir, f'{piece_name}.npz')
    original_wav_path = os.path.join(input_dir, f'{piece_name}.wav')

    if not os.path.exists(original_npz_path):
        print(f"Warning: {original_npz_path} not found, skipping")
        return []

    # Generate variants
    try:
        variants = generate_jump_sequences(
            input_dir, piece_name, scale_width, num_variants
        )
    except Exception as e:
        print(f"Error generating variants for {piece_name}: {e}")
        return []

    if not variants:
        return []

    generated_files = []

    for i, sequences in enumerate(variants):
        variant_name = f'{piece_name}_jump_v{i+1}'
        output_path = os.path.join(output_dir, f'{variant_name}.npz')

        save_jump_variant(original_npz_path, piece_name, sequences, output_path, scale_width)
        generated_files.append(output_path)

    # Create symlink for wav file (so loader can find it)
    if create_symlinks and os.path.exists(original_wav_path):
        symlink_path = os.path.join(output_dir, f'{piece_name}.wav')
        if not os.path.exists(symlink_path):
            try:
                # Use relative path for symlink
                rel_path = os.path.relpath(original_wav_path, output_dir)
                os.symlink(rel_path, symlink_path)
            except OSError as e:
                # Symlinks may not work on Windows without admin rights
                # Fall back to copying the file
                import shutil
                try:
                    shutil.copy2(original_wav_path, symlink_path)
                except Exception as copy_err:
                    print(f"Warning: Could not create symlink or copy {piece_name}.wav: {e}, {copy_err}")

    return generated_files


if __name__ == '__main__':
    import argparse
    import glob as glob_mod

    parser = argparse.ArgumentParser(
        description='Generate jump-augmented training data for score following'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing original .npz and .wav files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save jump-augmented files')
    parser.add_argument('--piece_name', type=str, default=None,
                        help='Single piece name to process (without extension). '
                             'If not specified, processes all .npz files in input_dir.')
    parser.add_argument('--num_variants', type=int, default=3,
                        help='Number of jump variants per piece (default: 3)')
    parser.add_argument('--scale_width', type=int, default=416,
                        help='Width to scale score images to (default: 416)')
    parser.add_argument('--no_symlinks', action='store_true',
                        help='Disable symlink/copy of wav files to output_dir')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.piece_name:
        piece_names = [args.piece_name]
    else:
        npz_files = sorted(glob_mod.glob(os.path.join(args.input_dir, '*.npz')))
        piece_names = [os.path.basename(f)[:-4] for f in npz_files]
        # Filter out existing jump variants
        piece_names = [p for p in piece_names if '_jump_v' not in p]

    if not piece_names:
        print(f"No .npz files found in {args.input_dir}")
        exit(1)

    print(f"Generating {args.num_variants} jump variant(s) for {len(piece_names)} piece(s)")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    total_generated = 0
    for piece_name in piece_names:
        print(f"\n  Processing: {piece_name}")
        generated = generate_and_save_jump_variants(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            piece_name=piece_name,
            scale_width=args.scale_width,
            num_variants=args.num_variants,
            create_symlinks=not args.no_symlinks,
        )
        total_generated += len(generated)
        for path in generated:
            print(f"    -> {os.path.basename(path)}")

    print(f"\nDone. Generated {total_generated} jump variant file(s) in {args.output_dir}")
