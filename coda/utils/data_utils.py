import cv2
import os

import numpy as np

from collections import Counter
from coda.utils.general import load_wav, xywh2xyxy
from scipy import interpolate


SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 1102
FPS = SAMPLE_RATE/HOP_SIZE


def load_piece(path, piece_name, audio_name=None):
    """
    Load a piece from .npz file.

    Args:
        path: Directory containing the .npz file
        piece_name: Name of the piece (without .npz extension)
        audio_name: Name of the audio file (without .wav extension).
                    If None, uses piece_name. Used for jump-augmented files
                    that reference the original audio.
    """
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)

    scores = npzfile["sheets"]
    coords, systems, bars = list(npzfile["coords"]), list(npzfile['systems']), list(npzfile['bars'])

    synthesized = npzfile['synthesized'].item()
    n_pages, h, w = scores.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((0, 0), (0, 0), (pad1, pad2))

    # Add padding
    padded_scores = np.pad(scores, pad, mode="constant", constant_values=255)

    # Use audio_name if provided (for jump-augmented files), else use piece_name
    if audio_name is None:
        audio_name = piece_name
    wav_path = os.path.join(path, audio_name + '.wav')
    signal = load_wav(wav_path, sr=SAMPLE_RATE)

    onsets = []
    for i in range(len(coords)):
        if coords[i]['note_x'] > 0:
            coords[i]['note_x'] += pad1

        # onset time to frame
        coords[i]['onset'] = int(coords[i]['onset'] * FPS)
        onsets.append(coords[i]['onset'])

    for i in range(len(systems)):
        systems[i]['x'] += pad1

    for i in range(len(bars)):
        bars[i]['x'] += pad1

    onsets = np.asarray(onsets, dtype=int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])
        bar_idx = int(Counter(onset_coords_merged['bar_idx']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        coords_new.append([note_y, note_x, system_idx, bar_idx, page_nr])
    coords_new = np.asarray(coords_new)

    # we want to match the frames to the coords of the previous onset, as the notes at the next coord position
    # aren't played yet
    interpol_fnc = interpolate.interp1d(onsets, coords_new.T, kind='previous', bounds_error=False,
                                        fill_value=(coords_new[0, :], coords_new[-1, :]))

    return padded_scores, scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad1, synthesized


def load_sequences(params):

    piece_idx = params['i']
    path = params['path']
    piece_name = params['piece_name']
    scale_width = params['scale_width']
    load_audio = params.get('load_audio', True)
    is_primary = params.get('is_primary', True)  # Whether to load scores/audio (False for jump variants)
    source_idx = params.get('source_idx', piece_idx)  # Which piece_id to use for scores/audio lookup

    # Check for audio_source and precomputed sequences in npzfile
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)

    # Determine audio source (for jump-augmented files, points to original piece)
    if 'audio_source' in npzfile:
        audio_name = str(npzfile['audio_source'].item())  # .item() for 0-dim array
    else:
        audio_name = piece_name

    # Check if precomputed sequences exist
    has_precomputed_sequences = 'sequences' in npzfile

    # For non-primary jump variants with precomputed sequences, only load sequences
    # This dramatically reduces memory usage (scores/audio shared with original)
    if not is_primary and has_precomputed_sequences:
        # Lightweight loading: only sequences, no scores/audio
        piece_sequences = list(npzfile['sequences'])

        # Update piece_id to point to the source piece (for score/audio lookup)
        for seq in piece_sequences:
            seq['piece_id'] = source_idx

        # Return minimal data - scores/audio/interpol will come from source piece
        npzfile.close()
        return (piece_idx, None, None, piece_name, piece_sequences, None, None, None, None)

    # Full loading for primary pieces (original behavior)
    padded_scores, _, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = \
        load_piece(path, piece_name, audio_name=audio_name)

    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    scale_factor = padded_scores.shape[1] / scale_width

    scaled_score = []
    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    scores = np.stack(scaled_score)

    duration = signal.shape[0]
    n_frames = int(np.ceil(FPS * duration / SAMPLE_RATE))
    piece_sequences = []

    interpol_c2o = {}
    staff_coords = {}
    add_per_staff = {}
    valid_pages = np.unique(coords_new[:, -1])

    for page_nr in valid_pages:
        page_coords = coords_new[coords_new[:, -1] == page_nr]
        page_onsets = onsets[coords_new[:, -1] == page_nr]
        unrolled_coords_x = []
        coords_per_staff = []

        # only add 0 for first staff
        max_xes = [0]
        staff_coords[page_nr] = sorted(np.unique(page_coords[:, 0]))

        for c in staff_coords[page_nr]:

            cs_staff = page_coords[page_coords[:, 0] == c, :-1]
            max_x = max(cs_staff[:, 1])
            coords_per_staff.append(cs_staff)
            max_xes.append(max_x)

        # last entry not needed
        add_per_staff[page_nr] = np.cumsum(max_xes)[:-1]
        for idx in range(len(staff_coords[page_nr])):
            unrolled_coords_x.append(coords_per_staff[idx][:, 1] + add_per_staff[page_nr][idx])

        unrolled_coords_x = np.concatenate(unrolled_coords_x)

        # Store raw arrays instead of interp1d (interp1d is not picklable with spawn)
        # Format: (sorted_coords, onsets, fill_value_left, fill_value_right)
        sort_idx = np.argsort(unrolled_coords_x)
        interpol_c2o[page_nr] = (
            unrolled_coords_x[sort_idx].astype(np.float32),
            page_onsets[sort_idx].astype(np.float32),
            float(page_onsets[0]),
            float(page_onsets[-1])
        )

    # Compute page-level layout metadata (for selection model + augmentation bounds)
    page_systems = {}
    page_bars = {}
    page_metadata = {}
    system_global_to_local = {}
    bar_global_to_local = {}

    for page_idx in valid_pages:
        page_systems[page_idx] = list(filter(lambda x: x['page_nr'] == page_idx, systems))
        page_bars[page_idx] = list(filter(lambda bar: bar['page_nr'] == page_idx, bars))

        p_sys = page_systems[page_idx]
        p_bar = page_bars[page_idx]

        sys_boxes_np = np.array([[s['x'], s['y'], s['w'], s['h']] for s in p_sys],
                                 dtype=np.float32) if p_sys else np.zeros((0, 4), dtype=np.float32)
        bar_boxes_np = np.array([[b['x'], b['y'], b['w'], b['h']] for b in p_bar],
                                 dtype=np.float32) if p_bar else np.zeros((0, 4), dtype=np.float32)

        # Map global system/bar indices to page-local indices
        page_sys_globals = [i for i, s in enumerate(systems) if s['page_nr'] == page_idx]
        page_bar_globals = [i for i, b in enumerate(bars) if b['page_nr'] == page_idx]
        for local_i, global_i in enumerate(page_sys_globals):
            system_global_to_local[global_i] = local_i
        for local_i, global_i in enumerate(page_bar_globals):
            bar_global_to_local[global_i] = local_i

        # bars_per_system via spatial containment (bar center within system bbox)
        bars_per_system = []
        for sys_dict in p_sys:
            sx1 = sys_dict['x'] - sys_dict['w'] / 2
            sy1 = sys_dict['y'] - sys_dict['h'] / 2
            sx2 = sys_dict['x'] + sys_dict['w'] / 2
            sy2 = sys_dict['y'] + sys_dict['h'] / 2
            member_bars = [bi for bi, bd in enumerate(p_bar)
                           if sx1 <= bd['x'] <= sx2 and sy1 <= bd['y'] <= sy2]
            bars_per_system.append(member_bars)

        page_metadata[page_idx] = {
            'system_boxes': sys_boxes_np,
            'bar_boxes': bar_boxes_np,
            'bars_per_system': bars_per_system,
        }

    # Check if we should use precomputed sequences
    if has_precomputed_sequences:
        # Use precomputed sequences (jump-augmented files)
        piece_sequences = list(npzfile['sequences'])

        # Optionally validate/recompute scale_factor to match current scale_width
        stored_scale_width = npzfile.get('scale_width', None)
        if stored_scale_width is not None:
            stored_sw = int(stored_scale_width.item()) if hasattr(stored_scale_width, 'item') else int(stored_scale_width)
            if stored_sw != scale_width:
                print(f"Warning: stored scale_width={stored_sw} != requested {scale_width}, recomputing scale_factor")
                for seq in piece_sequences:
                    seq['scale_factor'] = scale_factor

        # Update piece_id and add selection GT indices if missing
        n_invalid = 0
        for seq in piece_sequences:
            seq['piece_id'] = piece_idx
            if 'gt_system_page_idx' not in seq:
                tp = seq['true_position']
                s_idx, b_idx, pg = int(tp[2]), int(tp[3]), int(tp[-1])
                gt_sys_local = system_global_to_local.get(s_idx, None)
                gt_bar_local = bar_global_to_local.get(b_idx, None)

                # Validate system mapping
                if gt_sys_local is None:
                    seq['gt_valid'] = False
                    seq['gt_system_page_idx'] = 0
                    seq['gt_bar_in_system_idx'] = 0
                    n_invalid += 1
                    continue

                bps = page_metadata.get(pg, {}).get('bars_per_system', [])
                bars_in_sys = bps[gt_sys_local] if gt_sys_local < len(bps) else []

                # Validate bar-in-system mapping
                if gt_bar_local is None or gt_bar_local not in bars_in_sys:
                    seq['gt_valid'] = False
                    seq['gt_system_page_idx'] = gt_sys_local
                    seq['gt_bar_in_system_idx'] = 0
                    n_invalid += 1
                    continue

                seq['gt_valid'] = True
                seq['gt_system_page_idx'] = gt_sys_local
                seq['gt_bar_in_system_idx'] = bars_in_sys.index(gt_bar_local)

        if n_invalid > 0:
            print(f"  [data_utils] {piece_name}: {n_invalid}/{len(piece_sequences)} "
                  f"frames have invalid GT mapping ({n_invalid/len(piece_sequences)*100:.1f}%)")

        # Post-process: add temporal prior prev indices
        if not any('prev_gt_system_page_idx' in s for s in piece_sequences[:1]):
            prev_sys = -1
            prev_bar_page = -1
            prev_page = -1
            for seq in piece_sequences:
                pg = int(seq['true_position'][-1])
                if pg != prev_page:
                    prev_sys = -1
                    prev_bar_page = -1
                    prev_page = pg

                seq['prev_gt_system_page_idx'] = prev_sys
                seq['prev_gt_bar_page_idx'] = prev_bar_page

                if seq.get('gt_valid', True):
                    gt_sys = seq.get('gt_system_page_idx', 0)
                    gt_bar_in_sys = seq.get('gt_bar_in_system_idx', 0)
                    pg_meta = page_metadata.get(pg, {})
                    bps = pg_meta.get('bars_per_system', [])
                    if gt_sys < len(bps) and gt_bar_in_sys < len(bps[gt_sys]):
                        bar_page = bps[gt_sys][gt_bar_in_sys]
                        prev_sys = gt_sys
                        prev_bar_page = bar_page

    else:
        # Build sequences on-the-fly (original behavior)
        start_frame = 0
        curr_page = 0
        n_invalid = 0

        # Track previous frame's GT for temporal priors
        prev_sys_for_tp = -1    # -1 = no previous (first frame or page change)
        prev_bar_page_for_tp = -1  # page-local bar index of previous frame

        for frame in range(n_frames):

            true_position = np.asarray(interpol_fnc(frame), dtype=np.int32)

            bar_idx = true_position[3]
            system_idx = true_position[2]

            # figure out at which frame we change pages
            if true_position[-1] != curr_page:
                curr_page = true_position[-1]
                start_frame = frame
                # Reset temporal prior tracking on page change
                prev_sys_for_tp = -1
                prev_bar_page_for_tp = -1

            bar = bars[bar_idx]
            system = systems[system_idx]

            true_bar = np.asarray([bar['x'], bar['y'], bar['w'], bar['h']], dtype=float)
            true_system = np.asarray([system['x'], system['y'], system['w'], system['h']], dtype=float)

            systems_xywh = np.asarray([[x['x'], x['y'], x['w'], x['h']] for x in page_systems[curr_page]])
            systems_xyxy = xywh2xyxy(systems_xywh)

            max_x_shift = (-(int(systems_xyxy[:, 0].min() - 50)),
                           int(padded_scores.shape[2] - systems_xyxy[:, 2].max() - 50))
            max_y_shift = (min(0, -int((systems_xyxy[:, 1].min() - 50))),
                           max(1, int(padded_scores.shape[1] - systems_xyxy[:, 3].max() - 50)))

            # Selection model GT indices -- validate mappings
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

            piece_sequences.append({'piece_id': piece_idx,
                                    'is_onset': frame in onsets,
                                    'start_frame': start_frame,
                                    'frame': frame,
                                    'max_x_shift': max_x_shift,
                                    'max_y_shift': max_y_shift,
                                    'true_position': true_position,
                                    'true_system': true_system,
                                    'true_bar': true_bar,
                                    'height': system['h'],
                                    'synthesized': synthesized,
                                    'scale_factor': scale_factor,
                                    'gt_system_page_idx': gt_sys_local,
                                    'gt_bar_in_system_idx': gt_bar_in_sys,
                                    'gt_valid': gt_valid,
                                    'prev_gt_system_page_idx': prev_sys_for_tp,
                                    'prev_gt_bar_page_idx': prev_bar_page_for_tp,
                                    })

            # Update prev tracking for next frame (only if current is valid)
            if gt_valid:
                prev_sys_for_tp = gt_sys_local
                prev_bar_page_for_tp = gt_bar_local

        if n_invalid > 0:
            print(f"  [data_utils] {piece_name}: {n_invalid}/{n_frames} "
                  f"frames have invalid GT mapping ({n_invalid/n_frames*100:.1f}%)")

    if not load_audio:
        signal = os.path.join(path, audio_name + '.wav')

    return piece_idx, scores, signal, piece_name, piece_sequences, interpol_c2o, staff_coords, add_per_staff, page_metadata


def load_piece_for_testing(path, piece_name, scale_width):
    # Check for audio_source in npz (for jump-augmented files)
    npzfile = np.load(os.path.join(path, piece_name + '.npz'), allow_pickle=True)
    if 'audio_source' in npzfile:
        audio_name = str(npzfile['audio_source'].item())
    else:
        audio_name = None  # Will default to piece_name in load_piece
    npzfile.close()

    padded_scores, org_scores, onsets, _, bars, systems, interpol_fnc, signal, pad, _ = load_piece(path, piece_name, audio_name=audio_name)

    scores = 1 - np.array(padded_scores, dtype=np.float32) / 255.

    # scale scores
    scaled_score = []
    scale_factor = scores[0].shape[0] / scale_width

    for score in scores:
        scaled_score.append(cv2.resize(score, (scale_width, scale_width), interpolation=cv2.INTER_AREA))

    score = np.stack(scaled_score)

    org_scores_rgb = []
    for org_score in org_scores:
        org_score = np.array(org_score, dtype=np.float32) / 255.

        org_scores_rgb.append(cv2.cvtColor(org_score, cv2.COLOR_GRAY2BGR))

    return org_scores_rgb, score, signal, systems, bars, interpol_fnc, pad, scale_factor, onsets


def build_page_metadata(systems, bars):
    """
    Build per-page metadata from global systems/bars lists.

    Returns:
        page_meta: dict[page_nr] -> {
            'system_boxes': [N_sys, 4] xywh,
            'bar_boxes': [N_bar, 4] xywh,
            'bars_per_system': list of lists,
        }
    """
    all_pages = set(s['page_nr'] for s in systems)
    page_meta = {}

    for page_nr in all_pages:
        p_sys = [s for s in systems if s['page_nr'] == page_nr]
        p_bar = [b for b in bars if b['page_nr'] == page_nr]

        sys_boxes = np.array([[s['x'], s['y'], s['w'], s['h']] for s in p_sys],
                              dtype=np.float32) if p_sys else np.zeros((0, 4), dtype=np.float32)
        bar_boxes = np.array([[b['x'], b['y'], b['w'], b['h']] for b in p_bar],
                              dtype=np.float32) if p_bar else np.zeros((0, 4), dtype=np.float32)

        bars_per_system = []
        for sys_dict in p_sys:
            sx1 = sys_dict['x'] - sys_dict['w'] / 2
            sy1 = sys_dict['y'] - sys_dict['h'] / 2
            sx2 = sys_dict['x'] + sys_dict['w'] / 2
            sy2 = sys_dict['y'] + sys_dict['h'] / 2
            member_bars = [bi for bi, bd in enumerate(p_bar)
                           if sx1 <= bd['x'] <= sx2 and sy1 <= bd['y'] <= sy2]
            bars_per_system.append(member_bars)

        page_meta[page_nr] = {
            'system_boxes': sys_boxes,
            'bar_boxes': bar_boxes,
            'bars_per_system': bars_per_system,
        }

    return page_meta
