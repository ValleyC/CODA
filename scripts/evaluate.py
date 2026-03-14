"""
Inference script for Selection-Based Hierarchical Score Following.

Per-frame streaming inference:
  1. MambaConditioning.get_conditioning(spec_frame) -> z
  2. SharedBackbone(score_page, z) -> P3
  3. SelectionHead(P3, all_system_rois, z) -> log p(system_i)
  4. SelectionHead(P3, bars_in_top_systems, z) -> log p(bar_j | system_i)
  5. NoteHead(P3, selected_bar_roi, z) -> sigmoid (cx,cy)
  6. Beam search with temporal priors -> best path

Usage:
    python scripts/evaluate.py \
        --param_path path/to/checkpoint.pt \
        --test_dir path/to/test_data \
        --test_piece piece_name
"""

import cv2
import torch
import numpy as np
import matplotlib.cm as cm
from collections import deque
from tqdm import tqdm

from coda.models.coda_model import load_model
from coda.utils.data_utils import load_piece_for_testing, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from coda.utils.video_utils import create_video


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

    args = parser.parse_args()

    piece_name = args.test_piece
    org_scores, score, signal_np, systems, bars, interpol_fnc, pad, scale_factor, onsets = \
        load_piece_for_testing(args.test_dir, piece_name, args.scale_width)

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

    cond_net = network.conditioning_network
    signal = torch.from_numpy(signal_np).to(device)
    score_tensor = torch.from_numpy(score).unsqueeze(1).to(device)

    from_ = 0
    to_ = FRAME_SIZE
    hidden = None
    observation_images = []
    frame_idx = 0
    actual_page = 0
    track_page = args.page
    start_ = None
    vis_spec = None

    # Audio buffer for cross-attention in SelectionHeadV2 (system and/or bar head)
    cascade_cfg = network.yaml.get('cascade_config', {})
    sys_audio_window = cascade_cfg.get('head_v2', {}).get('audio_window', 64)
    bar_audio_window = cascade_cfg.get('bar_head_v2', {}).get('audio_window', 64)
    audio_window = max(sys_audio_window, bar_audio_window)
    audio_buffer = deque(maxlen=audio_window)

    frame_diffs = []
    total_frames = 0

    network.reset_tracking_state()

    pbar = tqdm(total=signal_np.shape[-1])

    while to_ <= signal_np.shape[-1]:
        true_position = np.array(interpol_fnc(frame_idx), dtype=np.float32)
        actual_system = int(true_position[2])
        actual_bar = int(true_position[3])

        if actual_page != int(true_position[-1]):
            hidden = None
            if hasattr(cond_net, 'reset_inference_state'):
                cond_net.reset_inference_state()
            network.reset_tracking_state()
            audio_buffer.clear()

        actual_page = int(true_position[-1])
        system = systems[actual_system]
        bar = bars[actual_bar]
        true_position_xy = true_position[:2]

        if track_page is None or actual_page == track_page:
            start_ = from_ if start_ is None else start_

            # Update break mode using waveform RMS (no GPU->CPU sync needed)
            break_diag = network.update_break_mode(signal_np[from_:to_])

            with torch.no_grad():
                sig_excerpt = signal[from_:to_]
                spec_frame = network.compute_spec([sig_excerpt], tempo_aug=False)[0]
                z, hidden = cond_net.get_conditioning(spec_frame, hidden=hidden)

                # Buffer Mamba output for cross-attention
                if hasattr(cond_net, 'get_cached_output'):
                    mamba_out = cond_net.get_cached_output()
                    if mamba_out is not None:
                        audio_buffer.append(mamba_out.squeeze(0))

                # Build audio_seq from buffer
                if len(audio_buffer) > 0:
                    audio_seq = torch.stack(list(audio_buffer), dim=0).unsqueeze(0)  # [1, T, 64]
                    audio_seq = audio_seq.to(device=device, dtype=z.dtype)
                    audio_lengths = torch.tensor([len(audio_buffer)], device=device)
                else:
                    audio_seq = None
                    audio_lengths = None

                p3, _ = network.backbone(
                    score_tensor[actual_page:actual_page + 1], z
                )

                # Get page metadata scaled to model input space
                pm = page_meta.get(actual_page)
                if pm is not None:
                    sys_boxes = torch.from_numpy(pm['system_boxes'] / scale_factor).to(device)
                    bar_boxes = torch.from_numpy(pm['bar_boxes'] / scale_factor).to(device)
                    bps = pm['bars_per_system']

                    result = network.inference_forward(
                        p3, z, sys_boxes, bar_boxes, bps,
                        audio_seq=audio_seq, audio_lengths=audio_lengths,
                    )
                else:
                    result = None

            # ── Visualization ────────────────────────────────────────────
            center_y, center_x = true_position_xy
            gt_x = center_x - pad
            gt_y = center_y

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

            if result is not None:
                pred_x = result['note_page_x'] * scale_factor - pad
                pred_y = result['note_page_y'] * scale_factor
                best = result['best_path']

                # -- Predicted system box --
                sys_idx = best['system_idx']
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
                bar_page_idx = best['bar_page_idx']
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

                # -- Compute error --
                frame_diff = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                frame_diffs.append(frame_diff)
                total_frames += 1

                # -- Info panel --
                info = {
                    'System': f"{sys_idx} (lp={best['sys_lp']:.2f})",
                    'Bar': f"{bar_page_idx} (lp={best['bar_lp']:.2f})",
                    'Error': f"{frame_diff:.1f} px",
                    'Mean Err': f"{np.mean(frame_diffs):.1f} px",
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
        else:
            if start_ is not None:
                break

        from_ += HOP_SIZE
        to_ += HOP_SIZE
        frame_idx += 1
        pbar.update(HOP_SIZE)

    pbar.close()

    if args.plot:
        cv2.destroyAllWindows()

    if frame_diffs:
        print(f"\n{'='*50}")
        print(f"[Results] {piece_name}")
        print(f"  Mean frame diff: {np.mean(frame_diffs):.2f} px")
        print(f"  Median frame diff: {np.median(frame_diffs):.2f} px")
        print(f"  Total frames: {total_frames}")
        print(f"{'='*50}")

    truncated_signal = signal_np[start_:to_]
    tag = "_selection" if args.page is None else f"_{args.page}_selection"
    create_video(observation_images, truncated_signal, piece_name, FPS, SAMPLE_RATE,
                 tag=tag, path=args.output_dir)
    print(f"Video saved to {args.output_dir}/{piece_name}{tag}.mp4")
