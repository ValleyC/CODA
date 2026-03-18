"""
Inference script for Selection-Based Hierarchical Score Following.

Per-frame streaming inference:
  1. MambaConditioning.get_conditioning(spec_frame) -> z
  2. SharedBackbone(score_page, z) -> P3
  3. SelectionHead(P3, all_system_rois, z) -> log p(system_i)
  4. SelectionHead(P3, bars_in_top_systems, z) -> log p(bar_j | system_i)
  5. NoteHead(P3, selected_bar_roi, z) -> sigmoid (cx,cy)
  6. Beam search with temporal priors -> best path

Supports both normal (linear) and jump-augmented test data.
For jump-augmented data, the NPZ contains a precomputed 'sequences' array
that defines the (possibly non-linear) frame order with silence insertions.

Usage:
    # Normal tracking
    python scripts/evaluate.py \
        --param_path path/to/checkpoint.pt \
        --test_dir path/to/test_data \
        --test_piece piece_name

    # Jump-augmented tracking
    python scripts/evaluate.py \
        --param_path path/to/checkpoint.pt \
        --test_dir path/to/test_data \
        --test_piece piece_name \
        --break_mode
"""

import os
import cv2
import time
import torch
import numpy as np
import matplotlib.cm as cm
from collections import deque
from tqdm import tqdm

from coda.models.coda_model import load_model
from coda.utils.data_utils import load_piece_for_testing, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from coda.utils.video_utils import create_video
import json


# ── Color palette (BGR uint8) -- muted tones, readable on white score ──────────
C_GT          = (30, 120, 200)     # muted rust/brown
C_GT_SYS      = (160, 130, 40)    # slate blue
C_GT_BAR      = (130, 50, 160)    # plum
C_PRED        = (180, 90, 40)     # steel blue
C_PRED_SYS    = (50, 160, 90)     # forest green
C_PRED_BAR    = (170, 120, 30)    # dark teal
C_INFO_BG     = (30, 30, 30)      # near-black


def overlay_box(img, x1, y1, x2, y2, color, alpha=0.15, border_thickness=2,
                label=None, label_scale=0.45):
    """Draw a semi-transparent filled box with solid border and label."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, border_thickness, cv2.LINE_AA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, label_scale, 1)
        lx, ly = x1, y1 - 4
        cv2.rectangle(img, (lx - 1, ly - th - 4), (lx + tw + 4, ly + 2), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (lx + 1, ly - 2), font, label_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_cursor(img, cx, sys_y1, sys_y2, color, dot_cy=None, radius=6,
                label=None, label_scale=0.45):
    """Draw a vertical playback cursor spanning the system with a glowing dot."""
    cx = int(cx)
    sy1, sy2 = int(sys_y1), int(sys_y2)

    # Translucent vertical highlight band (3px wide)
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - 1, sy1), (cx + 1, sy2), color, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Solid thin cursor line
    cv2.line(img, (cx, sy1), (cx, sy2), color, 1, cv2.LINE_AA)

    # Dot at note y position
    if dot_cy is not None:
        dot_cy = int(dot_cy)
        # Subtle outer ring
        cv2.circle(img, (cx, dot_cy), radius + 2, color, 1, cv2.LINE_AA)
        # Solid filled dot
        cv2.circle(img, (cx, dot_cy), radius, color, -1, cv2.LINE_AA)

    # Label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, label_scale, 1)
        lx = cx - tw // 2
        ly = sy1 - 6
        cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (lx, ly), font, label_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_panel(img, info_dict, panel_width=200):
    """Draw a translucent info panel on the top-right corner."""
    h = img.shape[0]
    panel_h = min(28 + len(info_dict) * 20, h)
    x_start = img.shape[1] - panel_width - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x_start, 4), (img.shape[1] - 4, panel_h + 4), C_INFO_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x_start, 4), (img.shape[1] - 4, panel_h + 4), (70, 70, 70), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Selection Model", (x_start + 6, 20), font, 0.40,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, (x_start + 4, 26), (img.shape[1] - 8, 26), (70, 70, 70), 1)
    y = 42
    for key, val in info_dict.items():
        cv2.putText(img, f"{key}:", (x_start + 6, y), font, 0.35,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, str(val), (x_start + 80, y), font, 0.35,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y += 20


def prepare_spec_panel(spec, score_h, width=200, colormap=cm.magma):
    """Render rolling spectrogram as a styled side panel."""
    if spec is None:
        return np.zeros((score_h, width, 3), dtype=np.float64)
    spec_flipped = np.flipud(spec)
    smin, smax = spec_flipped.min(), spec_flipped.max()
    if smax > smin:
        spec_norm = (spec_flipped - smin) / (smax - smin)
    else:
        spec_norm = np.zeros_like(spec_flipped)
    colored = colormap(spec_norm)[:, :, :3]
    colored_resized = cv2.resize(colored, (width - 20, score_h - 20))
    panel = np.zeros((score_h, width, 3), dtype=np.float64)
    panel[10:10 + colored_resized.shape[0], 10:10 + colored_resized.shape[1]] = colored_resized
    panel = panel[:, :, ::-1]  # RGB -> BGR
    return panel


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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Selection Inference')
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--test_piece', type=str, required=True)
    parser.add_argument('--scale_width', type=int, default=416)
    parser.add_argument('--page', type=int, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--output_dir', type=str, default='../videos')
    parser.add_argument('--bar_stay', type=float, default=None,
                        help='Override bar temporal prior: stay (default from config)')
    parser.add_argument('--bar_fwd1', type=float, default=None,
                        help='Override bar temporal prior: forward_1 (default from config)')
    parser.add_argument('--bar_fwd2', type=float, default=None,
                        help='Override bar temporal prior: forward_2 (default from config)')
    parser.add_argument('--bar_bwd1', type=float, default=None,
                        help='Override bar temporal prior: backward_1 (default from config)')
    parser.add_argument('--bar_far', type=float, default=None,
                        help='Override bar temporal prior: far (default from config)')
    # Break mode args
    parser.add_argument('--break_mode', action='store_true',
                        help='Enable break mode: silence-triggered prior suppression for jump recovery')
    parser.add_argument('--break_onset_threshold', type=float, default=0.1,
                        help='Break mode: normalized energy below this enters silence')
    parser.add_argument('--break_release_threshold', type=float, default=0.25,
                        help='Break mode: normalized energy above this exits silence')
    parser.add_argument('--break_silence_onset', type=int, default=3,
                        help='Break mode: consecutive silent frames before activating')
    parser.add_argument('--break_grace_frames', type=int, default=8,
                        help='Break mode: grace frames after audio resumes for relocalization')
    parser.add_argument('--break_prior_scale', type=float, default=0.0,
                        help='Break mode: temporal prior scaling (0 = full suppression)')
    parser.add_argument('--break_beam_k', type=int, default=-1,
                        help='Break mode: system beam width (-1 = all systems)')
    parser.add_argument('--break_beam_m', type=int, default=3,
                        help='Break mode: bar beam width during break')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (faster batch evaluation)')
    parser.add_argument('--save_metrics', type=str, default=None,
                        help='Save jump recovery metrics to JSON file')
    parser.add_argument('--benchmark', action='store_true',
                        help='Measure per-frame inference latency (ms) and report FPS')
    parser.add_argument('--benchmark_warmup', type=int, default=50,
                        help='Number of warmup frames to skip in benchmark timing')

    args = parser.parse_args()

    piece_name = args.test_piece

    # ── Load piece data ──────────────────────────────────────────────────
    org_scores, score, signal_np, systems, bars, interpol_fnc, pad, scale_factor, onsets = \
        load_piece_for_testing(args.test_dir, piece_name, args.scale_width)

    # Check for precomputed sequences (jump-augmented data)
    npz_path = os.path.join(args.test_dir, piece_name + '.npz')
    npzfile = np.load(npz_path, allow_pickle=True)
    has_sequences = 'sequences' in npzfile
    if has_sequences:
        sequences = list(npzfile['sequences'])
        jump_metadata = list(npzfile['jump_metadata']) if 'jump_metadata' in npzfile else []
        jump_output_indices = {int(j['output_idx']) for j in jump_metadata}
        # Pre-compute first non-silence frame after each jump (recovery window start)
        jump_dest_starts = []
        for jm in jump_metadata:
            oi = int(jm['output_idx'])
            dest = oi
            while dest < len(sequences) and sequences[dest].get('is_silence', False):
                dest += 1
            jump_dest_starts.append(dest)
        print(f"Jump-augmented mode: {len(sequences)} sequence entries, {len(jump_metadata)} jumps")
    npzfile.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading selection model from: {args.param_path}")
    network, criterion = load_model(args.param_path)

    # Override bar temporal priors if specified
    overrides = {
        'stay': args.bar_stay, 'forward_1': args.bar_fwd1,
        'forward_2': args.bar_fwd2, 'backward_1': args.bar_bwd1,
        'far': args.bar_far,
    }
    for key, val in overrides.items():
        if val is not None:
            network.bar_transition[key] = val
    print(f"Bar temporal priors: {network.bar_transition}")

    # Apply break mode config from CLI args
    if args.break_mode:
        network.break_mode_enabled = True
        network.break_onset_threshold = args.break_onset_threshold
        network.break_release_threshold = args.break_release_threshold
        network.break_silence_onset = args.break_silence_onset
        network.break_grace_frames = args.break_grace_frames
        network.break_prior_scale = args.break_prior_scale
        network.break_beam_k = args.break_beam_k
        network.break_beam_m = args.break_beam_m
        print(f"Break mode ENABLED: onset={args.break_onset_threshold}, "
              f"release={args.break_release_threshold}, "
              f"silence_onset={args.break_silence_onset}, "
              f"grace={args.break_grace_frames}, "
              f"prior_scale={args.break_prior_scale}")

    network.to(device)
    network.eval()
    print(f"Parameters: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}")

    # Build page metadata
    page_meta = build_page_metadata(systems, bars)

    # ── Index mappings for jump recovery metrics ──────────────────────
    # Global system idx -> page-local system idx
    global_to_page_sys = {}
    # (page_nr, page-local bar idx) -> global bar idx
    page_to_global_bar = {}
    for page_nr in page_meta:
        sys_on_page = [i for i, s in enumerate(systems) if s['page_nr'] == page_nr]
        for local_idx, global_idx in enumerate(sys_on_page):
            global_to_page_sys[global_idx] = local_idx
        bar_on_page = [i for i, b in enumerate(bars) if b['page_nr'] == page_nr]
        for local_idx, global_idx in enumerate(bar_on_page):
            page_to_global_bar[(page_nr, local_idx)] = global_idx

    # Bar onset time mapping: global bar_idx -> first onset time (seconds)
    bar_onset_time = {}
    for onset_frame in onsets:
        pos = interpol_fnc(onset_frame)
        bar_idx = int(pos[3])
        if bar_idx not in bar_onset_time:
            bar_onset_time[bar_idx] = onset_frame / FPS

    # ── Onset evaluation data (Table 1: onset error ratios) ───────
    onset_set = set(int(o) for o in onsets)
    onset_data = {}
    sys_onset_points = {}  # global_sys_idx -> {'xs': [], 'times': []}
    for onset_frame in onsets:
        pos = interpol_fnc(onset_frame)
        x_unpadded = pos[1] - pad
        sys_idx = int(pos[2])
        onset_data[int(onset_frame)] = {
            'x': x_unpadded,
            'time': onset_frame / FPS,
            'sys': sys_idx,
        }
        if sys_idx not in sys_onset_points:
            sys_onset_points[sys_idx] = {'xs': [], 'times': []}
        sys_onset_points[sys_idx]['xs'].append(x_unpadded)
        sys_onset_points[sys_idx]['times'].append(onset_frame / FPS)

    # Per-system inverse mapping: x_position -> time (seconds)
    system_x_to_time = {}
    for sys_idx, data in sys_onset_points.items():
        xs = np.array(data['xs'])
        ts = np.array(data['times'])
        _, unique_idx = np.unique(xs, return_index=True)
        xs, ts = xs[unique_idx], ts[unique_idx]
        sort_idx = np.argsort(xs)
        system_x_to_time[sys_idx] = (xs[sort_idx], ts[sort_idx])

    # (page_nr, page-local sys idx) -> global sys idx
    page_sys_to_global = {}
    for page_nr in page_meta:
        sys_on_page = [i for i, s in enumerate(systems) if s['page_nr'] == page_nr]
        for local_idx, global_idx in enumerate(sys_on_page):
            page_sys_to_global[(page_nr, local_idx)] = global_idx

    cond_net = network.conditioning_network
    signal = torch.from_numpy(signal_np).to(device)
    score_tensor = torch.from_numpy(score).unsqueeze(1).to(device)

    hidden = None
    observation_images = []
    actual_page = 0
    track_page = args.page
    vis_spec = None

    # Audio buffer for cross-attention in SelectionHeadV2 (system and/or bar head)
    cascade_cfg = network.yaml.get('cascade_config', {})
    sys_audio_window = cascade_cfg.get('head_v2', {}).get('audio_window', 64)
    bar_audio_window = cascade_cfg.get('bar_head_v2', {}).get('audio_window', 64)
    audio_window = max(sys_audio_window, bar_audio_window)
    audio_buffer = deque(maxlen=audio_window)

    frame_diffs = []
    total_frames = 0
    # For video audio track: collect signal chunks in playback order
    audio_chunks = []
    # Per-frame predictions for jump recovery metrics
    frame_records = {}
    # Standard tracking accuracy (all modes)
    sys_correct_count = 0
    bar_correct_count = 0
    total_eval_frames = 0
    # Onset predictions for Table 1 metrics (non-jump mode)
    onset_predictions = {}

    network.reset_tracking_state()

    # ── Benchmark timing ─────────────────────────────────────────────────
    _bench_times_total = []     # full pipeline per frame
    _bench_times_audio = []     # audio processing (spec + conditioning)
    _bench_times_backbone = []  # backbone forward
    _bench_times_heads = []     # cascade heads (system + bar + note)
    _bench_frame_counter = 0

    # ── Build frame iterator ─────────────────────────────────────────────
    # For jump-augmented data: iterate through precomputed sequences
    # For normal data: iterate linearly through audio frames
    if has_sequences:
        n_total = len(sequences)
        C_JUMP = (0, 0, 255)  # red for jump indicators

        def iter_frames():
            """Yield (audio_frame, true_position, is_silence, is_at_jump) per step."""
            for seq_idx, seq in enumerate(sequences):
                audio_frame = int(seq['frame'])
                true_pos = np.array(seq['true_position'], dtype=np.float32)
                is_silence = seq.get('is_silence', False)
                is_at_jump = seq_idx in jump_output_indices
                yield seq_idx, audio_frame, true_pos, is_silence, is_at_jump
    else:
        n_audio_frames = int(np.ceil(FPS * signal_np.shape[0] / SAMPLE_RATE))
        n_total = n_audio_frames

        def iter_frames():
            """Yield (audio_frame, true_position, is_silence, is_at_jump) per step."""
            for frame_idx in range(n_audio_frames):
                from_sample = frame_idx * HOP_SIZE
                to_sample = from_sample + FRAME_SIZE
                if to_sample > signal_np.shape[-1]:
                    break
                true_pos = np.array(interpol_fnc(frame_idx), dtype=np.float32)
                yield frame_idx, frame_idx, true_pos, False, False

    pbar = tqdm(total=n_total)
    prev_audio_frame = -1

    for step_idx, audio_frame, true_position, is_silence, is_at_jump in iter_frames():
        actual_system = int(true_position[2])
        actual_bar = int(true_position[3])
        new_page = int(true_position[-1])

        # ── State reset on page change ───────────────────────────────
        if actual_page != new_page:
            hidden = None
            if hasattr(cond_net, 'reset_inference_state'):
                cond_net.reset_inference_state()
            network.reset_tracking_state()
            audio_buffer.clear()

        # ── State reset at jump points (silence onset) ───────────────
        if is_at_jump:
            hidden = None
            if hasattr(cond_net, 'reset_inference_state'):
                cond_net.reset_inference_state()
            network.reset_tracking_state()
            audio_buffer.clear()

        actual_page = new_page
        system = systems[actual_system]
        bar = bars[actual_bar]
        true_position_xy = true_position[:2]

        # ── Compute audio boundaries ─────────────────────────────────
        from_ = audio_frame * HOP_SIZE
        to_ = from_ + FRAME_SIZE

        if track_page is not None and actual_page != track_page:
            pbar.update(1)
            continue

        # ── Feed audio / silence to model ────────────────────────────
        if is_silence or audio_frame < 0 or to_ > signal_np.shape[-1]:
            # Silence frame: zero audio
            sig_excerpt = torch.zeros(FRAME_SIZE, device=device)
            raw_audio = np.zeros(HOP_SIZE, dtype=np.float32)
            break_diag = network.update_break_mode(np.zeros(FRAME_SIZE))
        else:
            sig_excerpt = signal[from_:to_]
            raw_audio = signal_np[from_:from_ + HOP_SIZE]
            break_diag = network.update_break_mode(signal_np[from_:to_])

        _do_bench = args.benchmark and _bench_frame_counter >= args.benchmark_warmup

        if _do_bench:
            torch.cuda.synchronize()
            _t_total_start = time.perf_counter()

        with torch.no_grad():
            # --- Audio processing ---
            if _do_bench:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            spec_frame = network.compute_spec([sig_excerpt], tempo_aug=False)[0]
            z, hidden = cond_net.get_conditioning(spec_frame, hidden=hidden)

            # Buffer Mamba output for cross-attention
            if hasattr(cond_net, 'get_cached_output'):
                mamba_out = cond_net.get_cached_output()
                if mamba_out is not None:
                    audio_buffer.append(mamba_out.squeeze(0))

            # Build audio_seq from buffer
            if len(audio_buffer) > 0:
                audio_seq = torch.stack(list(audio_buffer), dim=0).unsqueeze(0)
                audio_seq = audio_seq.to(device=device, dtype=z.dtype)
                audio_lengths = torch.tensor([len(audio_buffer)], device=device)
            else:
                audio_seq = None
                audio_lengths = None

            if _do_bench:
                torch.cuda.synchronize()
                _bench_times_audio.append(time.perf_counter() - _t0)

            # --- Backbone ---
            if _do_bench:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            p3, _ = network.backbone(
                score_tensor[actual_page:actual_page + 1], z
            )

            if _do_bench:
                torch.cuda.synchronize()
                _bench_times_backbone.append(time.perf_counter() - _t0)

            # --- Cascade heads ---
            pm = page_meta.get(actual_page)
            if pm is not None:
                sys_boxes = torch.from_numpy(pm['system_boxes'] / scale_factor).to(device)
                bar_boxes = torch.from_numpy(pm['bar_boxes'] / scale_factor).to(device)
                bps = pm['bars_per_system']

                if _do_bench:
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()

                result = network.inference_forward(
                    p3, z, sys_boxes, bar_boxes, bps,
                    audio_seq=audio_seq, audio_lengths=audio_lengths,
                )

                if _do_bench:
                    torch.cuda.synchronize()
                    _bench_times_heads.append(time.perf_counter() - _t0)
            else:
                result = None

        if _do_bench:
            torch.cuda.synchronize()
            _bench_times_total.append(time.perf_counter() - _t_total_start)

        _bench_frame_counter += 1

        # ── Computation: extract predictions and record metrics ─────
        center_y, center_x = true_position_xy
        gt_x = center_x - pad
        gt_y = center_y

        frame_diff = None
        if result is not None:
            pred_x = result['note_page_x'] * scale_factor - pad
            pred_y = result['note_page_y'] * scale_factor
            best = result['best_path']
            sys_idx = best['system_idx']
            bar_page_idx = best['bar_page_idx']

            gt_sys_local = global_to_page_sys.get(actual_system, -1)
            pred_bar_global = page_to_global_bar.get(
                (actual_page, bar_page_idx), -1)

            # Record for jump recovery metrics
            if has_sequences:
                frame_records[step_idx] = {
                    'pred_sys': sys_idx,
                    'gt_sys': gt_sys_local,
                    'pred_bar_global': pred_bar_global,
                    'gt_bar_global': actual_bar,
                    'is_silence': is_silence,
                }

            # Compute error (skip silence frames)
            if not is_silence:
                frame_diff = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                frame_diffs.append(frame_diff)
                total_frames += 1

                # System & bar accuracy (Table 1)
                if sys_idx == gt_sys_local:
                    sys_correct_count += 1
                if pred_bar_global == actual_bar:
                    bar_correct_count += 1
                total_eval_frames += 1

                # Store onset prediction (non-jump mode, for Table 1 error ratios)
                if not has_sequences and audio_frame in onset_set:
                    pred_sys_global = page_sys_to_global.get(
                        (actual_page, sys_idx), -1)
                    onset_predictions[audio_frame] = {
                        'pred_x': pred_x,
                        'pred_sys_global': pred_sys_global,
                    }

        # ── Visualization (skip if --no_video) ────────────────────────
        do_video = not args.no_video
        if do_video:
            audio_chunks.append(raw_audio)

            # Start from clean score image (uint8 BGR)
            img = (cv2.cvtColor(org_scores[actual_page], cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)

            # System extent for cursor lines
            s = system
            sys_y1 = s['y'] - s['h'] / 2
            sys_y2 = s['y'] + s['h'] / 2

            # -- GT system box (semi-transparent) --
            sx1 = s['x'] - s['w'] / 2 - pad
            sx2 = s['x'] + s['w'] / 2 - pad
            overlay_box(img, sx1, sys_y1, sx2, sys_y2, C_GT_SYS, alpha=0.06,
                        border_thickness=1, label="GT System")

            # -- GT bar box (semi-transparent) --
            b = bar
            bx1, by1 = b['x'] - b['w'] / 2 - pad, b['y'] - b['h'] / 2
            bx2, by2 = b['x'] + b['w'] / 2 - pad, b['y'] + b['h'] / 2
            overlay_box(img, bx1, by1, bx2, by2, C_GT_BAR, alpha=0.10,
                        border_thickness=1, label="GT Bar")

            # -- GT cursor: vertical line + glowing dot --
            draw_cursor(img, gt_x, sys_y1, sys_y2, C_GT, dot_cy=gt_y,
                        radius=5, label="GT")

            # -- Jump/silence indicator --
            if has_sequences and is_silence:
                cv2.putText(img, "SILENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, C_JUMP, 2, cv2.LINE_AA)
            if has_sequences and is_at_jump:
                cv2.putText(img, "JUMP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, C_JUMP, 2, cv2.LINE_AA)

            if result is not None:
                # -- Predicted system box --
                pred_sys_y1, pred_sys_y2 = sys_y1, sys_y2  # fallback
                if sys_idx < pm['system_boxes'].shape[0]:
                    sb = pm['system_boxes'][sys_idx]
                    psx1 = sb[0] - sb[2] / 2 - pad
                    psy1 = sb[1] - sb[3] / 2
                    psx2 = sb[0] + sb[2] / 2 - pad
                    psy2 = sb[1] + sb[3] / 2
                    pred_sys_y1, pred_sys_y2 = psy1, psy2
                    overlay_box(img, psx1, psy1, psx2, psy2, C_PRED_SYS, alpha=0.08,
                                border_thickness=2,
                                label=f"Pred Sys {sys_idx} ({best['sys_lp']:.1f})")

                # -- Predicted bar box --
                if bar_page_idx < pm['bar_boxes'].shape[0]:
                    bb = pm['bar_boxes'][bar_page_idx]
                    pbx1 = bb[0] - bb[2] / 2 - pad
                    pby1 = bb[1] - bb[3] / 2
                    pbx2 = bb[0] + bb[2] / 2 - pad
                    pby2 = bb[1] + bb[3] / 2
                    overlay_box(img, pbx1, pby1, pbx2, pby2, C_PRED_BAR, alpha=0.10,
                                border_thickness=2,
                                label=f"Pred Bar {bar_page_idx} ({best['bar_lp']:.1f})")

                # -- Predicted cursor: vertical line + glowing dot --
                draw_cursor(img, pred_x, pred_sys_y1, pred_sys_y2, C_PRED,
                            dot_cy=pred_y, radius=7, label="Pred")

                # -- Info panel --
                info = {
                    'System': f"{sys_idx} (lp={best['sys_lp']:.2f})",
                    'Bar': f"{bar_page_idx} (lp={best['bar_lp']:.2f})",
                    'Error': f"{frame_diff:.1f} px" if frame_diff is not None else "—",
                    'Mean Err': f"{np.mean(frame_diffs):.1f} px" if frame_diffs else "—",
                    'Frame': f"{total_frames}",
                }
                # Break mode diagnostics
                if network.break_mode_enabled:
                    ne = break_diag['norm_energy']
                    ne_str = 'warmup' if ne < 0 else f"{ne:.2f}"
                    brk_status = 'SILENCE' if break_diag['in_silence'] else (
                        f"GRACE({break_diag['grace_frames_remaining']})"
                        if break_diag['is_break_mode'] else 'off')
                    info['Break'] = brk_status
                    info['Energy'] = ne_str
                draw_info_panel(img, info)

            # -- Rolling spectrogram side panel --
            if vis_spec is not None:
                vis_spec = np.roll(vis_spec, -1, axis=1)
            else:
                vis_spec = np.zeros((spec_frame.shape[-1], 60))
            vis_spec[:, -1] = spec_frame[0].cpu().numpy()

            spec_panel = prepare_spec_panel(vis_spec, img.shape[0], width=220)
            spec_panel = (spec_panel * 255).astype(np.uint8)
            img = np.concatenate((img, spec_panel), axis=1)

            if args.plot:
                cv2.imshow('Selection Prediction', img)
                cv2.waitKey(20)

            observation_images.append(img)

        prev_audio_frame = audio_frame
        pbar.update(1)

    pbar.close()

    if args.plot:
        cv2.destroyAllWindows()

    # ── Basic tracking results ────────────────────────────────────────
    if frame_diffs:
        print(f"\n{'='*50}")
        print(f"[Results] {piece_name}")
        print(f"  Mean frame diff: {np.mean(frame_diffs):.2f} px")
        print(f"  Median frame diff: {np.median(frame_diffs):.2f} px")
        print(f"  Total frames: {total_frames}")
        if has_sequences:
            print(f"  Jumps: {len(jump_metadata)}")
        print(f"{'='*50}")

    # ── Standard Tracking Metrics (Table 1) ────────────────────────────
    ONSET_THRESHOLDS = [0.05, 0.10, 0.50, 1.00, 5.00]  # seconds
    sys_accuracy = sys_correct_count / total_eval_frames if total_eval_frames else 0
    bar_accuracy = bar_correct_count / total_eval_frames if total_eval_frames else 0

    onset_errors = []
    if not has_sequences and onset_predictions:
        for onset_frame in onsets:
            of = int(onset_frame)
            pred = onset_predictions.get(of)
            if pred is None:
                continue
            od = onset_data[of]
            gt_time = od['time']
            pred_sys_global = pred['pred_sys_global']
            pred_x_val = pred['pred_x']

            mapping = system_x_to_time.get(pred_sys_global)
            if mapping is not None and len(mapping[0]) >= 2:
                xs, ts = mapping
                pred_time = float(np.interp(pred_x_val, xs, ts))
                time_err = abs(pred_time - gt_time)
            else:
                time_err = 100.0  # wrong system or sparse mapping
            onset_errors.append(time_err)

    onset_ratios = {}
    if onset_errors:
        for t in ONSET_THRESHOLDS:
            onset_ratios[t] = sum(1 for e in onset_errors if e <= t) / len(onset_errors)

        print(f"\n{'='*60}")
        print(f"[Standard Tracking Metrics] {piece_name}")
        print(f"{'='*60}")
        print(f"  System accuracy: {sys_accuracy:.4f}")
        print(f"  Bar accuracy:    {bar_accuracy:.4f}")
        print(f"  Onsets evaluated: {len(onset_errors)}")
        print(f"  --- Onset Error Ratios ---")
        for t in ONSET_THRESHOLDS:
            print(f"    <={t:.2f}s: {onset_ratios[t]:.4f}")
        print(f"{'='*60}")
    elif total_eval_frames > 0:
        print(f"\n  System accuracy: {sys_accuracy:.4f}  "
              f"Bar accuracy: {bar_accuracy:.4f}")

    # ── Build base metrics dict (always saved) ─────────────────────────
    metrics = {
        'piece': piece_name,
        'sys_accuracy': sys_accuracy,
        'bar_accuracy': bar_accuracy,
        'total_eval_frames': total_eval_frames,
        'mean_frame_diff_px': float(np.mean(frame_diffs)) if frame_diffs else None,
    }
    for t in ONSET_THRESHOLDS:
        metrics[f'onset_ratio_{t:.2f}s'] = onset_ratios.get(t, None)
    if onset_errors:
        metrics['n_onsets_evaluated'] = len(onset_errors)
        metrics['mean_onset_err_s'] = float(np.mean(onset_errors))
        metrics['median_onset_err_s'] = float(np.median(onset_errors))

    # ── Jump Recovery Metrics (comprehensive) ───────────────────────────
    # System recovery rate at multiple thresholds, tracking error at
    # multiple tolerances, evaluated in a post-jump window.
    RECOVERY_THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0]   # seconds
    ERROR_THRESHOLDS    = [0.5, 1.0, 2.0, 3.0]          # seconds
    POST_JUMP_WINDOW    = 5.0                            # seconds
    if has_sequences and frame_records and jump_metadata:
        window_frames = {t: int(round(t * FPS)) for t in RECOVERY_THRESHOLDS}
        post_window = int(round(POST_JUMP_WINDOW * FPS))

        # Per-jump accumulators
        recovery_lists = {t: [] for t in RECOVERY_THRESHOLDS}
        latency_list = []
        # Post-jump tracking: collect raw time errors for all frames
        post_jump_time_errors = []

        for ji, dest_start in enumerate(jump_dest_starts):
            recovered = {t: False for t in RECOVERY_THRESHOLDS}
            first_correct_offset = None

            max_window = max(max(window_frames.values()), post_window)
            for offset in range(max_window):
                idx = dest_start + offset
                rec = frame_records.get(idx)
                if rec is None or rec['is_silence']:
                    continue

                sys_correct = (rec['pred_sys'] == rec['gt_sys'])

                if sys_correct and first_correct_offset is None:
                    first_correct_offset = offset

                for t in RECOVERY_THRESHOLDS:
                    if offset < window_frames[t] and sys_correct:
                        recovered[t] = True

                # Post-jump tracking error: bar onset time difference
                if offset < post_window:
                    gt_time = bar_onset_time.get(rec['gt_bar_global'], 0)
                    pred_time = bar_onset_time.get(rec['pred_bar_global'], 0)
                    time_err = abs(pred_time - gt_time)
                    post_jump_time_errors.append(time_err)

            for t in RECOVERY_THRESHOLDS:
                recovery_lists[t].append(recovered[t])
            if first_correct_offset is not None:
                latency_list.append(first_correct_offset / FPS)
            else:
                latency_list.append(None)

        n_jumps = len(jump_dest_starts)
        recovery_rates = {}
        for t in RECOVERY_THRESHOLDS:
            recovery_rates[t] = sum(recovery_lists[t]) / n_jumps if n_jumps else 0
        valid_latencies = [l for l in latency_list if l is not None]
        mean_lat = np.mean(valid_latencies) if valid_latencies else float('inf')
        median_lat = np.median(valid_latencies) if valid_latencies else float('inf')

        error_accuracies = {}
        for t in ERROR_THRESHOLDS:
            error_accuracies[t] = (
                sum(1 for e in post_jump_time_errors if e <= t)
                / len(post_jump_time_errors)
            ) if post_jump_time_errors else 0

        # Print comprehensive results
        print(f"\n{'='*60}")
        print(f"[Jump Recovery Metrics] {piece_name}")
        print(f"{'='*60}")
        print(f"  Jumps evaluated: {n_jumps}")
        print(f"  --- System Recovery Rate ---")
        for t in RECOVERY_THRESHOLDS:
            n_rec = sum(recovery_lists[t])
            print(f"    @{t:.1f}s: {recovery_rates[t]:.3f}  ({n_rec}/{n_jumps})")
        lat_str = f"{mean_lat:.3f}" if mean_lat < float('inf') else "N/A"
        med_str = f"{median_lat:.3f}" if median_lat < float('inf') else "N/A"
        print(f"  --- Recovery Latency ---")
        print(f"    Mean:   {lat_str} s  "
              f"(recovered: {len(valid_latencies)}/{n_jumps})")
        print(f"    Median: {med_str} s")
        print(f"  --- Post-Jump Tracking Accuracy ({POST_JUMP_WINDOW:.0f}s window) ---")
        for t in ERROR_THRESHOLDS:
            n_ok = sum(1 for e in post_jump_time_errors if e <= t)
            print(f"    Err<={t:.1f}s: {error_accuracies[t]:.3f}  "
                  f"({n_ok}/{len(post_jump_time_errors)} frames)")
        if post_jump_time_errors:
            print(f"    Mean err: {np.mean(post_jump_time_errors):.3f} s  "
                  f"Median: {np.median(post_jump_time_errors):.3f} s")
        print(f"{'='*60}")

        metrics['n_jumps'] = n_jumps
        metrics['mean_latency_s'] = mean_lat if mean_lat < float('inf') else None
        metrics['median_latency_s'] = median_lat if median_lat < float('inf') else None
        metrics['n_recovered'] = len(valid_latencies)
        metrics['latency_per_jump'] = latency_list
        metrics['post_jump_total_frames'] = len(post_jump_time_errors)
        for t in RECOVERY_THRESHOLDS:
            metrics[f'sys_recovery_{t:.1f}s'] = recovery_rates[t]
            metrics[f'n_recovered_{t:.1f}s'] = sum(recovery_lists[t])
        for t in ERROR_THRESHOLDS:
            metrics[f'post_jump_err_{t:.1f}s'] = error_accuracies[t]
            metrics[f'post_jump_ok_{t:.1f}s'] = sum(
                1 for e in post_jump_time_errors if e <= t)
        if post_jump_time_errors:
            metrics['post_jump_mean_err_s'] = float(np.mean(post_jump_time_errors))
            metrics['post_jump_median_err_s'] = float(np.median(post_jump_time_errors))

    # ── Save metrics (always) ──────────────────────────────────────────
    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics) or '.', exist_ok=True)
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.save_metrics}")

    # ── Benchmark results ─────────────────────────────────────────────
    if args.benchmark and _bench_times_total:
        n = len(_bench_times_total)
        total_ms = np.array(_bench_times_total) * 1000
        audio_ms = np.array(_bench_times_audio) * 1000 if _bench_times_audio else np.array([0])
        backbone_ms = np.array(_bench_times_backbone) * 1000 if _bench_times_backbone else np.array([0])
        heads_ms = np.array(_bench_times_heads) * 1000 if _bench_times_heads else np.array([0])

        mean_total = total_ms.mean()
        fps = 1000.0 / mean_total if mean_total > 0 else float('inf')

        print(f"\n{'='*60}")
        print(f"[Benchmark] {piece_name}")
        print(f"{'='*60}")
        print(f"  Frames timed: {n} (after {args.benchmark_warmup} warmup)")
        print(f"  --- Per-frame latency (ms) ---")
        print(f"    Total:    {mean_total:.2f} mean | {np.median(total_ms):.2f} median | {np.std(total_ms):.2f} std")
        print(f"    Audio:    {audio_ms.mean():.2f} mean | {np.median(audio_ms):.2f} median")
        print(f"    Backbone: {backbone_ms.mean():.2f} mean | {np.median(backbone_ms):.2f} median")
        print(f"    Heads:    {heads_ms.mean():.2f} mean | {np.median(heads_ms):.2f} median")
        print(f"  --- Throughput ---")
        print(f"    {fps:.1f} FPS (mean)  |  {1000.0/np.median(total_ms):.1f} FPS (median)")
        print(f"    Real-time factor: {fps / FPS:.2f}x (target: {FPS} FPS)")
        print(f"  --- Percentiles (total, ms) ---")
        for p in [50, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(total_ms, p):.2f}")
        print(f"{'='*60}")

        # Add to metrics dict
        metrics['benchmark'] = {
            'n_frames': n,
            'warmup_frames': args.benchmark_warmup,
            'mean_total_ms': float(mean_total),
            'median_total_ms': float(np.median(total_ms)),
            'std_total_ms': float(np.std(total_ms)),
            'mean_audio_ms': float(audio_ms.mean()),
            'mean_backbone_ms': float(backbone_ms.mean()),
            'mean_heads_ms': float(heads_ms.mean()),
            'fps_mean': float(fps),
            'fps_median': float(1000.0 / np.median(total_ms)),
            'realtime_factor': float(fps / FPS),
            'p50_ms': float(np.percentile(total_ms, 50)),
            'p90_ms': float(np.percentile(total_ms, 90)),
            'p95_ms': float(np.percentile(total_ms, 95)),
            'p99_ms': float(np.percentile(total_ms, 99)),
        }

        # Re-save metrics with benchmark data
        if args.save_metrics:
            with open(args.save_metrics, 'w') as f:
                json.dump(metrics, f, indent=2)

    # ── Video generation ──────────────────────────────────────────────
    if not args.no_video:
        if audio_chunks:
            video_signal = np.concatenate(audio_chunks)
        else:
            video_signal = signal_np

        tag = "_selection" if args.page is None else f"_{args.page}_selection"
        if has_sequences:
            tag = "_jump" + tag
        create_video(observation_images, video_signal, piece_name, FPS, SAMPLE_RATE,
                     tag=tag, path=args.output_dir)
        print(f"Video saved to {args.output_dir}/{piece_name}{tag}.mp4")
