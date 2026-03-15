"""
Batch evaluation for score following.

Evaluates all pieces in a directory, aggregates per-piece metrics,
and prints summary tables suitable for the ISMIR manuscript.

Supports two evaluation modes:
  - Standard tracking (Table 1): onset error ratios, system/bar accuracy
  - Jump recovery (Table 2): recovery rate, latency, post-jump tracking

Usage:
    # Standard tracking on MSMD test set (Table 1)
    python scripts/evaluate_batch.py \
        --param_path pretrained/best_model.pt \
        --test_dir data/msmd/msmd_test \
        --label "CODA (ours)" \
        --metrics_dir results/metrics/standard \
        --save_summary results/standard_summary.json

    # Jump recovery on repeat subset (Table 2)
    python scripts/evaluate_batch.py \
        --param_path pretrained/best_model.pt \
        --test_dir data/msmd/msmd_test_jump/repeat \
        --break_mode --label "CODA (full) - repeat" \
        --metrics_dir results/metrics/repeat \
        --save_summary results/repeat_summary.json

    # With videos (inline, one piece at a time)
    python scripts/evaluate_batch.py \
        --param_path pretrained/best_model.pt \
        --test_dir data/msmd/msmd_test_jump/repeat \
        --break_mode --label "CODA (full) repeat" \
        --save_summary results/repeat_summary.json \
        --with_video --video_dir results/videos/repeat
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np
from collections import defaultdict


def find_pieces(test_dir):
    """Find all NPZ files in test_dir and return piece names."""
    npz_files = sorted(glob.glob(os.path.join(test_dir, '*.npz')))
    return [os.path.splitext(os.path.basename(f))[0] for f in npz_files]


def run_evaluate(piece_name, args, metrics_path):
    """Run evaluate.py for a single piece, return metrics dict or None."""
    cmd = [
        sys.executable, 'scripts/evaluate.py',
        '--param_path', args.param_path,
        '--test_dir', args.test_dir,
        '--test_piece', piece_name,
        '--save_metrics', metrics_path,
    ]
    if args.with_video:
        cmd.extend(['--output_dir', args.video_dir])
    else:
        cmd.append('--no_video')
    if args.break_mode:
        cmd.append('--break_mode')
        if args.break_onset_threshold is not None:
            cmd.extend(['--break_onset_threshold', str(args.break_onset_threshold)])
        if args.break_release_threshold is not None:
            cmd.extend(['--break_release_threshold', str(args.break_release_threshold)])
        if args.break_silence_onset is not None:
            cmd.extend(['--break_silence_onset', str(args.break_silence_onset)])
        if args.break_grace_frames is not None:
            cmd.extend(['--break_grace_frames', str(args.break_grace_frames)])
        if args.break_prior_scale is not None:
            cmd.extend(['--break_prior_scale', str(args.break_prior_scale)])
        if args.break_beam_k is not None:
            cmd.extend(['--break_beam_k', str(args.break_beam_k)])
        if args.break_beam_m is not None:
            cmd.extend(['--break_beam_m', str(args.break_beam_m)])

    # Let stderr (tqdm progress bar) pass through to terminal
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {piece_name}")
        return None

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None



def aggregate_metrics(all_metrics):
    """Aggregate per-piece metrics into summary statistics.

    Handles both standard tracking (Table 1) and jump recovery (Table 2)
    metrics. Only aggregates fields that are present in the per-piece data.
    """
    if not all_metrics:
        return {}

    summary = {}
    n_pieces = len(all_metrics)
    summary['n_pieces'] = n_pieces

    # ── Standard tracking metrics (Table 1) ────────────────────────
    # System and bar accuracy (always present)
    sys_accs = [m['sys_accuracy'] for m in all_metrics
                if m.get('sys_accuracy') is not None]
    bar_accs = [m['bar_accuracy'] for m in all_metrics
                if m.get('bar_accuracy') is not None]
    if sys_accs:
        summary['mean_sys_accuracy'] = float(np.mean(sys_accs))
    if bar_accs:
        summary['mean_bar_accuracy'] = float(np.mean(bar_accs))

    # Onset error ratios (non-jump mode only)
    ONSET_THRESHOLDS = [0.05, 0.10, 0.50, 1.00, 5.00]
    for t in ONSET_THRESHOLDS:
        key = f'onset_ratio_{t:.2f}s'
        ratios = [m[key] for m in all_metrics if m.get(key) is not None]
        if ratios:
            summary[f'mean_{key}'] = float(np.mean(ratios))

    # ── Jump recovery metrics (Table 2) ────────────────────────────
    has_jumps = any(m.get('n_jumps') is not None for m in all_metrics)
    if has_jumps:
        total_jumps = sum(m.get('n_jumps', 0) for m in all_metrics)
        summary['total_jumps'] = total_jumps

        # System recovery rates (macro + micro)
        recovery_keys = [k for k in all_metrics[0] if k.startswith('sys_recovery_')]
        for key in recovery_keys:
            rates = [m[key] for m in all_metrics if key in m]
            summary[f'macro_{key}'] = np.mean(rates) if rates else 0
            n_key = key.replace('sys_recovery_', 'n_recovered_')
            n_rec = sum(m.get(n_key, 0) for m in all_metrics)
            summary[f'micro_{key}'] = n_rec / total_jumps if total_jumps else 0
            summary[f'total_{n_key}'] = n_rec

        # Recovery latency
        all_latencies = []
        for m in all_metrics:
            if m.get('latency_per_jump'):
                all_latencies.extend([l for l in m['latency_per_jump'] if l is not None])
        summary['mean_latency_s'] = float(np.mean(all_latencies)) if all_latencies else None
        summary['median_latency_s'] = float(np.median(all_latencies)) if all_latencies else None
        summary['n_recovered_total'] = len(all_latencies)

        # Post-jump tracking accuracy
        error_keys = [k for k in all_metrics[0] if k.startswith('post_jump_err_')]
        for key in error_keys:
            rates = [m[key] for m in all_metrics if key in m]
            summary[f'macro_{key}'] = np.mean(rates) if rates else 0
            ok_key = key.replace('post_jump_err_', 'post_jump_ok_')
            n_ok = sum(m.get(ok_key, 0) for m in all_metrics)
            total_frames = sum(m.get('post_jump_total_frames', 0) for m in all_metrics)
            summary[f'micro_{key}'] = n_ok / total_frames if total_frames else 0
        summary['post_jump_total_frames'] = sum(
            m.get('post_jump_total_frames', 0) for m in all_metrics)

        # Post-jump mean/median error
        mean_errs = [m['post_jump_mean_err_s'] for m in all_metrics
                     if m.get('post_jump_mean_err_s') is not None]
        summary['mean_post_jump_err_s'] = float(np.mean(mean_errs)) if mean_errs else None

    # Pixel-level frame diff
    diffs = [m['mean_frame_diff_px'] for m in all_metrics
             if m.get('mean_frame_diff_px') is not None]
    summary['mean_frame_diff_px'] = float(np.mean(diffs)) if diffs else None

    return summary


def print_summary(summary, label=""):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Pieces: {summary.get('n_pieces', 0)}   "
          f"Total jumps: {summary.get('total_jumps', 0)}")

    print(f"\n  {'System Recovery Rate':>30}  {'Macro':>8}  {'Micro':>8}")
    print(f"  {'-'*50}")
    for t in [0.5, 1.0, 2.0, 3.0, 5.0]:
        key = f'sys_recovery_{t:.1f}s'
        macro = summary.get(f'macro_{key}', 0)
        micro = summary.get(f'micro_{key}', 0)
        n_rec = summary.get(f'total_n_recovered_{t:.1f}s', 0)
        print(f"  {'@' + f'{t:.1f}s':>30}  {macro:>7.3f}  {micro:>7.3f}  "
              f"({n_rec}/{summary.get('total_jumps', 0)})")

    print(f"\n  {'Recovery Latency':>30}")
    print(f"  {'-'*50}")
    lat = summary.get('mean_latency_s')
    med = summary.get('median_latency_s')
    n_rec = summary.get('n_recovered_total', 0)
    n_tot = summary.get('total_jumps', 0)
    print(f"  {'Mean':>30}  {f'{lat:.3f} s' if lat else 'N/A':>8}  "
          f"(recovered: {n_rec}/{n_tot})")
    print(f"  {'Median':>30}  {f'{med:.3f} s' if med else 'N/A':>8}")

    print(f"\n  {'Post-Jump Tracking Accuracy':>30}  {'Macro':>8}  {'Micro':>8}")
    print(f"  {'-'*50}")
    for t in [0.5, 1.0, 2.0, 3.0]:
        key = f'post_jump_err_{t:.1f}s'
        macro = summary.get(f'macro_{key}', 0)
        micro = summary.get(f'micro_{key}', 0)
        print(f"  {'Err<=' + f'{t:.1f}s':>30}  {macro:>7.3f}  {micro:>7.3f}")
    mean_err = summary.get('mean_post_jump_err_s')
    if mean_err is not None:
        print(f"  {'Mean tracking err':>30}  {mean_err:.3f} s")

    if summary.get('mean_frame_diff_px') is not None:
        print(f"\n  {'Mean frame diff':>30}  {summary['mean_frame_diff_px']:.1f} px")
    print(f"{'='*70}")


def print_latex_row(summary, label="CODA (full)"):
    """Print a LaTeX table row for Table 2 (jump recovery)."""
    rec_1 = summary.get('micro_sys_recovery_1.0s', 0)
    rec_2 = summary.get('micro_sys_recovery_2.0s', 0)
    lat = summary.get('mean_latency_s')
    err_1 = summary.get('micro_post_jump_err_1.0s', 0)
    lat_str = f"{lat:.2f}" if lat is not None else "--"
    print(f"\n  LaTeX row (Table 2):")
    print(f"  {label} & {rec_1:.3f} & {rec_2:.3f} & {lat_str} & {err_1:.3f} \\\\")


def print_tracking_summary(summary, label=""):
    """Print Table 1 standard tracking metrics."""
    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Pieces: {summary.get('n_pieces', 0)}")

    # Onset error ratios
    has_onsets = any(summary.get(f'mean_onset_ratio_{t:.2f}s') is not None
                     for t in [0.05, 0.10, 0.50, 1.00, 5.00])
    if has_onsets:
        print(f"\n  {'Error Threshold':>25}  {'Ratio':>8}")
        print(f"  {'-'*40}")
        for t in [0.05, 0.10, 0.50, 1.00, 5.00]:
            key = f'mean_onset_ratio_{t:.2f}s'
            ratio = summary.get(key)
            if ratio is not None:
                print(f"  {'<=' + f'{t:.2f}s':>25}  {ratio:>7.4f}")

    # System and bar accuracy
    sys_acc = summary.get('mean_sys_accuracy')
    bar_acc = summary.get('mean_bar_accuracy')
    if sys_acc is not None or bar_acc is not None:
        print(f"\n  {'Accuracy':>25}  {'Rate':>8}")
        print(f"  {'-'*40}")
        if sys_acc is not None:
            print(f"  {'System':>25}  {sys_acc:>7.4f}")
        if bar_acc is not None:
            print(f"  {'Bar':>25}  {bar_acc:>7.4f}")

    if summary.get('mean_frame_diff_px') is not None:
        print(f"\n  {'Mean frame diff':>25}  {summary['mean_frame_diff_px']:.1f} px")
    print(f"{'='*70}")


def print_tracking_latex_row(summary, label="CODA (ours)"):
    """Print a LaTeX table row for Table 1 (standard tracking)."""
    thresholds = [0.05, 0.10, 0.50, 1.00, 5.00]
    values = []
    for t in thresholds:
        v = summary.get(f'mean_onset_ratio_{t:.2f}s')
        values.append(f"{v:.3f}" if v is not None else "--")
    bar_acc = summary.get('mean_bar_accuracy')
    sys_acc = summary.get('mean_sys_accuracy')
    values.append(f"{bar_acc:.3f}" if bar_acc is not None else "--")
    values.append(f"{sys_acc:.3f}" if sys_acc is not None else "--")
    print(f"\n  LaTeX row (Table 1):")
    print(f"  {label} & {' & '.join(values)} \\\\")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch evaluation (standard tracking and/or jump recovery)')
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--label', type=str, default='',
                        help='Label for this evaluation run')
    parser.add_argument('--save_summary', type=str, default=None,
                        help='Save aggregated summary to JSON')
    parser.add_argument('--metrics_dir', type=str, default='results/metrics',
                        help='Directory for per-piece metric JSONs')
    # Break mode
    parser.add_argument('--break_mode', action='store_true')
    parser.add_argument('--break_onset_threshold', type=float, default=None)
    parser.add_argument('--break_release_threshold', type=float, default=None)
    parser.add_argument('--break_silence_onset', type=int, default=None)
    parser.add_argument('--break_grace_frames', type=int, default=None)
    parser.add_argument('--break_prior_scale', type=float, default=None)
    parser.add_argument('--break_beam_k', type=int, default=None)
    parser.add_argument('--break_beam_m', type=int, default=None)
    # Video
    parser.add_argument('--with_video', action='store_true',
                        help='Also generate videos inline (one piece at a time)')
    parser.add_argument('--video_dir', type=str, default='results/videos',
                        help='Output directory for videos')
    parser.add_argument('--pieces', type=str, nargs='+', default=None,
                        help='Only evaluate these specific pieces (by name)')

    args = parser.parse_args()

    pieces = find_pieces(args.test_dir)
    if args.pieces:
        piece_set = set(args.pieces)
        pieces = [p for p in pieces if p in piece_set]
    if not pieces:
        print(f"No NPZ files found in {args.test_dir}")
        sys.exit(1)

    print(f"Found {len(pieces)} pieces in {args.test_dir}")
    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── Phase 1: Sequential metrics evaluation (GPU-bound) ────────────
    all_metrics = []
    failed = []
    for i, piece in enumerate(pieces):
        print(f"[{i+1}/{len(pieces)}] {piece}")
        metrics_path = os.path.join(args.metrics_dir, f"{piece}.json")
        m = run_evaluate(piece, args, metrics_path)
        if m is not None:
            all_metrics.append(m)
        else:
            failed.append(piece)

    # ── Phase 2: Aggregate and print ──────────────────────────────────
    summary = aggregate_metrics(all_metrics)

    # Print tracking metrics (Table 1) — always available
    print_tracking_summary(summary, label=args.label)
    print_tracking_latex_row(summary, label=args.label or "CODA (ours)")

    # Print jump recovery metrics (Table 2) — only if jump data
    if summary.get('total_jumps'):
        print_summary(summary, label=args.label)
        print_latex_row(summary, label=args.label or "Method")

    if failed:
        print(f"\n  Failed pieces ({len(failed)}):")
        for p in failed:
            print(f"    - {p}")

    if args.save_summary:
        os.makedirs(os.path.dirname(args.save_summary) or '.', exist_ok=True)
        # Convert numpy types for JSON serialization
        clean = {}
        for k, v in summary.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v
        clean['label'] = args.label
        clean['failed_pieces'] = failed
        with open(args.save_summary, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Summary saved to {args.save_summary}")
