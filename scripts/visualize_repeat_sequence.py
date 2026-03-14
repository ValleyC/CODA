"""
Visualize a repeat-augmented sequence as a video for verification.

Renders the score page with highlighted system/bar and a playback cursor,
synchronized with the original audio. No model inference needed.

Follows the same rendering pattern as score_transformer/scripts/visualize_jump_data.py.

Usage:
    python scripts/visualize_repeat_sequence.py \
        --npz_path data/msmd/msmd_test_repeat/repeat/BachJS__BWV994__bach-applicatio_synth.npz \
        --audio_dir data/msmd/msmd_test \
        --output_dir videos/verify
"""

import os
import argparse
import cv2
import numpy as np
import soundfile as sf
import tempfile
import time

from tqdm import tqdm

from coda.utils.data_utils import load_piece, SAMPLE_RATE, FPS, HOP_SIZE, FRAME_SIZE
from coda.utils.video_utils import plot_box, plot_line, mux_video_audio
from coda.utils.general import xywh2xyxy


def detect_jumps(sequences):
    """Detect jump points in the sequence (where frame number decreases or jumps)."""
    jumps = []
    for i in range(1, len(sequences)):
        prev_frame = sequences[i-1]['frame']
        curr_frame = sequences[i]['frame']
        if curr_frame != prev_frame + 1:
            jumps.append({
                'seq_idx': i,
                'from_frame': prev_frame,
                'to_frame': curr_frame,
                'jump_type': 'backward' if curr_frame < prev_frame else 'forward'
            })
    return jumps


def build_spliced_audio(sequences, signal, hop_size=HOP_SIZE):
    """Build audio matching the sequence order (one hop per frame).
    Silence frames (marked is_silence=True) are zeroed out."""
    audio_segments = []
    for seq in sequences:
        if seq.get('is_silence', False):
            audio_segments.append(np.zeros(hop_size))
            continue
        frame = seq['frame']
        start_sample = frame * hop_size
        end_sample = start_sample + hop_size
        if end_sample <= len(signal):
            audio_segments.append(signal[start_sample:end_sample])
        else:
            segment = np.zeros(hop_size)
            valid_len = len(signal) - start_sample
            if valid_len > 0:
                segment[:valid_len] = signal[start_sample:start_sample + valid_len]
            audio_segments.append(segment)
    return np.concatenate(audio_segments)


def main():
    parser = argparse.ArgumentParser(description='Visualize repeat-augmented sequence')
    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to the generated repeat-augmented NPZ file')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing original WAV files')
    parser.add_argument('--output_dir', type=str, default='videos/verify')
    parser.add_argument('--scale_width', type=int, default=416)
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the generated NPZ to get sequences and audio_source
    npzfile = np.load(args.npz_path, allow_pickle=True)
    sequences = list(npzfile['sequences'])
    audio_source = str(npzfile['audio_source'])
    piece_name = os.path.basename(args.npz_path).replace('.npz', '')

    print(f"Piece: {piece_name}")
    print(f"Audio source: {audio_source}")
    print(f"Sequence length: {len(sequences)} frames")

    # Load piece using load_piece (which handles padding + coordinate offsets)
    piece_dir = os.path.dirname(args.npz_path)
    padded_scores, org_scores, onsets, coords_new, bars, systems, interpol_fnc, signal, pad, synthesized = \
        load_piece(piece_dir, piece_name, audio_name=audio_source)

    # Convert org_scores to float BGR for visualization
    org_scores_rgb = []
    for org_score in org_scores:
        org_score_float = np.array(org_score, dtype=np.float32) / 255.
        org_scores_rgb.append(cv2.cvtColor(org_score_float, cv2.COLOR_GRAY2BGR))

    print(f"Score pages: {len(org_scores_rgb)}, size: {org_scores_rgb[0].shape}")
    print(f"Systems: {len(systems)}, Bars: {len(bars)}, Pad: {pad}")
    print(f"Signal: {len(signal)} samples ({len(signal)/SAMPLE_RATE:.1f}s)")

    # Detect jumps
    jumps = detect_jumps(sequences)
    print(f"\nDetected {len(jumps)} jumps:")
    for j in jumps:
        print(f"  Seq idx {j['seq_idx']}: frame {j['from_frame']} -> {j['to_frame']} ({j['jump_type']})")

    # Limit frames if specified
    if args.max_frames is not None:
        sequences = sequences[:args.max_frames]

    # Build spliced audio
    print("\nBuilding spliced audio...")
    spliced_audio = build_spliced_audio(sequences, signal)
    wav_path = os.path.join(tempfile.gettempdir(), f'{time.time()}.wav')
    sf.write(wav_path, spliced_audio, samplerate=SAMPLE_RATE)
    del spliced_audio

    # Setup video writer
    height_px, width_px = org_scores_rgb[0].shape[:2]
    video_path = os.path.join(tempfile.gettempdir(), f'{time.time()}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width_px, height_px))

    jump_indices = {j['seq_idx'] for j in jumps}
    jump_flash_countdown = 0

    print("\nRendering frames...")
    for seq_idx, seq in enumerate(tqdm(sequences, desc="Rendering")):
        frame = seq['frame']
        page_nr = int(seq['true_position'][-1])
        system_idx = int(seq['true_position'][2])
        bar_idx = int(seq['true_position'][3])
        center_y = seq['true_position'][0]
        center_x = seq['true_position'][1]

        system = systems[system_idx]
        bar = bars[bar_idx]
        height = system['h'] / 2

        # Create image (same as visualize_jump_data.py)
        img = cv2.cvtColor(org_scores_rgb[page_nr].copy(), cv2.COLOR_RGB2BGR)

        # Jump flash
        is_jump = seq_idx in jump_indices
        if is_jump:
            jump_flash_countdown = 10

        if jump_flash_countdown > 0:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0.8), -1)
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "JUMP!"
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            cv2.putText(img, text, (text_x, 55), font, 2, (1, 1, 1), 3, cv2.LINE_AA)
            jump_flash_countdown -= 1

        # Draw position line (orange) - subtract pad from x
        if center_x > 0:
            plot_line([center_x - pad, center_y, height], img,
                      color=(0.96, 0.63, 0.25), line_thickness=3, label=f"Frame {frame}")

        # Draw system box (blue) - subtract pad from x
        plot_box(xywh2xyxy(np.asarray([[system['x'] - pad, system['y'],
                                        system['w'], system['h']]]))[0].astype(int).tolist(),
                 img, color=(0.25, 0.71, 0.96), line_thickness=2, label=f"System {system_idx}")

        # Draw bar box (pink) - subtract pad from x
        plot_box(xywh2xyxy(np.asarray([[bar['x'] - pad, bar['y'],
                                        bar['w'], bar['h']]]))[0].astype(int).tolist(),
                 img, color=(0.96, 0.24, 0.69), line_thickness=2, label=f"Bar {bar_idx}")

        # Info text at bottom
        info_y = img.shape[0] - 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"Seq: {seq_idx} | Frame: {frame} | Bar: {bar_idx} | Sys: {system_idx} | Page: {page_nr}"
        cv2.putText(img, info_text, (10, info_y), font, 0.6, (0.2, 0.2, 0.2), 2, cv2.LINE_AA)
        cv2.putText(img, info_text, (10, info_y), font, 0.6, (1, 1, 1), 1, cv2.LINE_AA)

        # Write frame
        img = np.array((img * 255), dtype=np.uint8)
        video_writer.write(img)

    video_writer.release()

    # Mux video and audio
    print("\nMuxing video and audio...")
    output_path = os.path.join(args.output_dir, f'{piece_name}_verify.mp4')
    mux_video_audio(video_path, wav_path, path_output=output_path)

    os.remove(video_path)
    os.remove(wav_path)

    print(f"\nVideo saved to: {output_path}")
    print(f"\nJump summary:")
    for j in jumps:
        print(f"  At seq {j['seq_idx']}: frame {j['from_frame']} -> {j['to_frame']}")


if __name__ == '__main__':
    main()
