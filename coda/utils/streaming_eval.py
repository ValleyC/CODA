"""
Streaming evaluation for the selection cascade model.

Runs per-frame streaming inference and computes online accuracy metrics.
Used during training for model selection based on real streaming
performance rather than teacher-forced validation loss.
"""

import os
from collections import deque
import torch
import numpy as np
from tqdm import tqdm

from coda.utils.data_utils import (
    load_piece_for_testing, build_page_metadata,
    SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
)


def streaming_eval_piece(network, test_dir, piece_name, scale_width, device,
                         verbose=False):
    """
    Run streaming inference on a single piece and return metrics.

    Args:
        network: SelectionCascadeModel (already on device, eval mode)
        test_dir: path to directory containing piece NPZ + WAV
        piece_name: name of piece (without extension)
        scale_width: model input width (e.g. 416)
        device: torch device
        verbose: if True, print per-frame progress

    Returns:
        dict with keys:
            'n_frames': total frames evaluated
            'sys_correct': number of system-correct frames
            'bar_correct': number of bar-correct frames (page-local bar index)
            'pixel_errors': list of per-frame pixel errors
            'sys_acc': system accuracy
            'bar_acc': bar accuracy
            'mean_px_err': mean pixel error
            'median_px_err': median pixel error
    """
    org_scores, score, signal_np, systems, bars, interpol_fnc, pad, scale_factor, onsets = \
        load_piece_for_testing(test_dir, piece_name, scale_width)

    page_meta = build_page_metadata(systems, bars)

    # Build global-to-page-local index maps (same logic as data_utils.py)
    sys_global_to_local = {}
    bar_global_to_local = {}
    for page_nr in page_meta:
        page_sys_globals = [i for i, s in enumerate(systems) if s['page_nr'] == page_nr]
        page_bar_globals = [i for i, b in enumerate(bars) if b['page_nr'] == page_nr]
        for local_i, global_i in enumerate(page_sys_globals):
            sys_global_to_local[global_i] = local_i
        for local_i, global_i in enumerate(page_bar_globals):
            bar_global_to_local[global_i] = local_i

    cond_net = network.conditioning_network
    signal = torch.from_numpy(signal_np).to(device)
    score_tensor = torch.from_numpy(score).unsqueeze(1).to(device)

    from_ = 0
    to_ = FRAME_SIZE
    hidden = None
    frame_idx = 0
    actual_page = 0

    # Audio buffer for cross-attention in SelectionHeadV2 (system and/or bar head)
    cascade_cfg = network.yaml.get('cascade_config', {})
    sys_audio_window = cascade_cfg.get('head_v2', {}).get('audio_window', 64)
    bar_audio_window = cascade_cfg.get('bar_head_v2', {}).get('audio_window', 64)
    audio_window = max(sys_audio_window, bar_audio_window)
    audio_buffer = deque(maxlen=audio_window)

    n_frames = 0
    sys_correct = 0
    bar_correct = 0
    pixel_errors = []

    network.reset_tracking_state()

    pbar = tqdm(total=signal_np.shape[-1], disable=not verbose,
                desc=piece_name[:30], leave=False)

    while to_ <= signal_np.shape[-1]:
        true_position = np.array(interpol_fnc(frame_idx), dtype=np.float32)
        gt_system_global = int(true_position[2])
        gt_bar_global = int(true_position[3])
        current_page = int(true_position[-1])

        # Reset on page change
        if actual_page != current_page:
            hidden = None
            if hasattr(cond_net, 'reset_inference_state'):
                cond_net.reset_inference_state()
            network.reset_tracking_state()
            audio_buffer.clear()

        actual_page = current_page

        # Update break mode using waveform RMS (no GPU->CPU sync needed)
        network.update_break_mode(signal_np[from_:to_])

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

        if result is not None:
            best = result['best_path']
            pred_sys = best['system_idx']
            pred_bar_page = best['bar_page_idx']

            # GT page-local indices
            gt_sys_local = sys_global_to_local.get(gt_system_global, -1)
            gt_bar_local = bar_global_to_local.get(gt_bar_global, -1)

            # System accuracy
            if pred_sys == gt_sys_local:
                sys_correct += 1

            # Bar accuracy (page-local bar index)
            if pred_bar_page == gt_bar_local:
                bar_correct += 1

            # Pixel error
            pred_x = result['note_page_x'] * scale_factor - pad
            pred_y = result['note_page_y'] * scale_factor
            gt_y, gt_x = true_position[:2]
            gt_x = gt_x - pad

            px_err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            pixel_errors.append(px_err)

            n_frames += 1

        from_ += HOP_SIZE
        to_ += HOP_SIZE
        frame_idx += 1
        pbar.update(HOP_SIZE)

    pbar.close()

    sys_acc = sys_correct / max(n_frames, 1)
    bar_acc = bar_correct / max(n_frames, 1)
    px_arr = np.array(pixel_errors) if pixel_errors else np.array([0.0])

    return {
        'n_frames': n_frames,
        'sys_correct': sys_correct,
        'bar_correct': bar_correct,
        'pixel_errors': pixel_errors,
        'sys_acc': sys_acc,
        'bar_acc': bar_acc,
        'mean_px_err': float(px_arr.mean()),
        'median_px_err': float(np.median(px_arr)),
    }


def streaming_eval(network, test_dir, piece_names, scale_width, device,
                   verbose=True):
    """
    Run streaming evaluation on multiple pieces.

    Args:
        network: SelectionCascadeModel (will be set to eval mode)
        test_dir: path to directory containing piece NPZ + WAV files
        piece_names: list of piece names to evaluate
        scale_width: model input width (e.g. 416)
        device: torch device
        verbose: if True, print per-piece results

    Returns:
        dict with aggregate metrics:
            'sys_acc': overall system accuracy
            'bar_acc': overall bar accuracy
            'mean_px_err': overall mean pixel error
            'median_px_err': overall median pixel error
            'n_frames': total frames across all pieces
            'per_piece': dict[piece_name] -> per-piece metrics
    """
    network.eval()

    total_frames = 0
    total_sys_correct = 0
    total_bar_correct = 0
    all_pixel_errors = []
    per_piece = {}

    for piece_name in piece_names:
        try:
            metrics = streaming_eval_piece(
                network, test_dir, piece_name, scale_width, device,
                verbose=False
            )
        except Exception as e:
            if verbose:
                print(f"  [streaming_eval] ERROR on {piece_name}: {e}")
            continue

        per_piece[piece_name] = metrics
        total_frames += metrics['n_frames']
        total_sys_correct += metrics['sys_correct']
        total_bar_correct += metrics['bar_correct']
        all_pixel_errors.extend(metrics['pixel_errors'])

        if verbose:
            print(f"  {piece_name}: sys={metrics['sys_acc']:.3f} "
                  f"bar={metrics['bar_acc']:.3f} "
                  f"px_err={metrics['mean_px_err']:.1f} "
                  f"({metrics['n_frames']} frames)")

    sys_acc = total_sys_correct / max(total_frames, 1)
    bar_acc = total_bar_correct / max(total_frames, 1)
    px_arr = np.array(all_pixel_errors) if all_pixel_errors else np.array([0.0])

    result = {
        'sys_acc': sys_acc,
        'bar_acc': bar_acc,
        'mean_px_err': float(px_arr.mean()),
        'median_px_err': float(np.median(px_arr)),
        'n_frames': total_frames,
        'per_piece': per_piece,
    }

    if verbose:
        print(f"\n  [streaming_eval] AGGREGATE: sys={sys_acc:.3f} bar={bar_acc:.3f} "
              f"px_err={result['mean_px_err']:.1f} "
              f"({total_frames} frames, {len(per_piece)} pieces)")

    return result
