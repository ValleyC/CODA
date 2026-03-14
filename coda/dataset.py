
import glob
import os
import random
import torch
import torchvision

import numpy as np


from coda.utils.data_utils import load_sequences, SAMPLE_RATE, FPS, FRAME_SIZE, HOP_SIZE
from coda.utils.general import load_wav, load_yaml
from coda.augmentations.impulse_response import ImpulseResponse

from multiprocessing import get_context
from torch.utils.data import Dataset
from tqdm import tqdm

# Musically realistic jump type distribution for on-the-fly augmentation
# Weighted to match real-world score jump frequency
JUMP_TYPE_WEIGHTS = {
    'repeat': 0.40,        # Jump backward 1-2 systems (repeat signs, D.C.) — most common
    'skip': 0.15,          # Jump forward 1-3 systems (D.S. al Coda, skip)
    'restart': 0.10,       # Jump to first system on page (D.C. al Fine)
    'page_jump': 0.10,     # Cross-page jump (D.S., Coda on different page)
    'bar_correction': 0.15, # Different bar in same/adjacent system (small repeat/correction)
    'random': 0.10,        # Arbitrary system (stress test, catch-all)
}

# Align jump augmentation with break mode defaults:
# - at least 3 silent frames to trigger break onset
# - only a short post-silence window for relocalization
JUMP_SILENCE_MIN_FRAMES = 3
JUMP_SILENCE_MAX_FRAMES = 12
JUMP_POST_SILENCE_MIN_FRAMES = 1
JUMP_POST_SILENCE_MAX_FRAMES = 8


def nearest_interp(x, interpol_data):
    """
    Nearest-neighbor interpolation using stored arrays (picklable replacement for interp1d).

    Args:
        x: Query point(s)
        interpol_data: Tuple of (sorted_coords, onsets, fill_left, fill_right)

    Returns:
        Interpolated onset value(s)
    """
    coords, onsets, fill_left, fill_right = interpol_data
    x = np.atleast_1d(x)

    # Find nearest index using searchsorted
    idx = np.searchsorted(coords, x)

    # Handle boundary cases
    idx = np.clip(idx, 0, len(coords) - 1)

    # Check if we should use left or right neighbor
    left_idx = np.maximum(idx - 1, 0)

    # Use left neighbor if it's closer
    use_left = (idx > 0) & (np.abs(coords[left_idx] - x) < np.abs(coords[idx] - x))
    idx = np.where(use_left, left_idx, idx)

    result = onsets[idx]

    # Apply fill values for out-of-bounds
    result = np.where(x < coords[0], fill_left, result)
    result = np.where(x > coords[-1], fill_right, result)

    return result[0] if result.size == 1 else result


class SequenceDataset(Dataset):
    def __init__(self, scores, performances, sequences, piece_names, interpol_c2o,
                 staff_coords, add_per_staff, predict_sb=False, system_only=False,
                 augment=False, transform=None, cold_start_prob=0.0, cold_start_min_frames=50,
                 cold_start_min_context=20, cold_start_max_context=400, page_metadata=None,
                 jump_prob=0.0):
        """
        Args:
            scores: Dict of score images per piece
            performances: Dict of audio signals per piece
            sequences: List of frame sequence dicts
            piece_names: Dict of piece names
            interpol_c2o: Interpolation data for coordinate-to-onset mapping
            staff_coords: Staff coordinate data
            add_per_staff: Additional per-staff data
            predict_sb: Whether to predict system and bar
            system_only: System-only mode
            augment: Enable spatial/audio augmentation
            transform: Additional transforms
            cold_start_prob: Probability of cold-starting (resetting audio context).
                            0.0 = always use full accumulated context (default)
                            0.5 = 50% use cold start, 50% use full context
                            This acts as regularization to prevent overfitting to
                            trajectory-based predictions.
            cold_start_min_frames: Minimum frame number to allow cold start.
                                   Frames before this always use normal context.
                                   Default 50 (2.5 sec) to ensure some warm-up data exists.
            cold_start_min_context: Minimum context length when cold-starting (in frames).
                                    Default 20 (~1 sec at 20 FPS).
            cold_start_max_context: Maximum context length when cold-starting (in frames).
                                    Default 400 (~20 sec at 20 FPS).
                                    Random length sampled uniformly between min and max.
            page_metadata: Dict of page layout metadata per piece (for selection model).
            jump_prob: Probability of on-the-fly jump augmentation per sample.
                       When triggered, swaps destination to a different system/bar
                       while keeping the source's prev_gt (biased temporal priors).
                       Teaches model to re-localize from audio evidence.
        """
        self.scores = scores
        self.performances = performances
        self.sequences = []
        self.rand_perf_indices = {}
        self.sequences = sequences
        self.augment = augment
        self.piece_names = piece_names
        self.interpol_c2o = interpol_c2o
        self.staff_coords = staff_coords
        self.add_per_staff = add_per_staff
        self.page_metadata = page_metadata

        self.predict_sb = predict_sb
        self.system_only = system_only

        # Cold-start regularization parameters
        self.cold_start_prob = cold_start_prob
        self.cold_start_min_frames = cold_start_min_frames
        self.cold_start_min_context = cold_start_min_context
        self.cold_start_max_context = cold_start_max_context

        # Jump augmentation parameters
        self.jump_prob = jump_prob
        if self.jump_prob > 0:
            self._build_jump_indices()

        self.fps = FPS
        self.sample_rate = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_SIZE

        self.length = len(self.sequences)
        self.gt_width = 30
        self.transform = transform

        # Explicit augmentation status
        aug_status = "ENABLED (spatial shifts + audio padding)" if self.augment else "DISABLED"
        print(f"[Dataset] Augmentation: {aug_status}")
        if self.cold_start_prob > 0:
            print(f"[Dataset] Cold-start regularization: {self.cold_start_prob*100:.0f}% prob, "
                  f"min_frames={self.cold_start_min_frames}, "
                  f"context={self.cold_start_min_context}-{self.cold_start_max_context} frames "
                  f"({self.cold_start_min_context/self.fps:.1f}-{self.cold_start_max_context/self.fps:.1f} sec)")
        if self.jump_prob > 0:
            print(f"[Dataset] Jump augmentation: {self.jump_prob*100:.0f}% prob, "
                  f"{len(JUMP_TYPE_WEIGHTS)} jump types (onset-biased destinations)")

    def __len__(self):
        return self.length

    def _build_jump_indices(self):
        """Build multi-level indices for structured jump destination sampling."""
        self._piece_indices = {}      # piece_id -> [seq indices]
        self._piece_system_map = {}   # piece_id -> {(page, sys_page_idx) -> [seq indices]}
        self._piece_page_map = {}     # piece_id -> {page -> [seq indices]}

        for i, seq in enumerate(self.sequences):
            pid = seq['piece_id']
            page = int(seq['true_position'][-1])
            sys_idx = seq.get('gt_system_page_idx', 0)

            if pid not in self._piece_indices:
                self._piece_indices[pid] = []
                self._piece_system_map[pid] = {}
                self._piece_page_map[pid] = {}

            self._piece_indices[pid].append(i)

            key = (page, sys_idx)
            if key not in self._piece_system_map[pid]:
                self._piece_system_map[pid][key] = []
            self._piece_system_map[pid][key].append(i)

            if page not in self._piece_page_map[pid]:
                self._piece_page_map[pid][page] = []
            self._piece_page_map[pid][page].append(i)

    def _sample_jump_destination(self, item):
        """Sample a destination for jump augmentation with structured jump types.

        Jump types and what they model:
          repeat (40%):        Backward 1-2 systems (70%) or 3-4 (30%)
          skip (15%):          Forward 1-3 systems
          restart (10%):       First system on current page
          page_jump (10%):     Different page entirely
          bar_correction (15%): Different bar in same/adjacent system
          random (10%):        Arbitrary different system

        Destinations are biased toward onset frames (note attacks / bar boundaries).
        """
        seq = self.sequences[item]
        piece_id = seq['piece_id']
        src_page = int(seq['true_position'][-1])
        src_sys = seq.get('gt_system_page_idx', 0)
        src_bar = seq.get('gt_bar_in_system_idx', 0)

        # Sample jump type from weighted distribution
        r = random.random()
        cumsum = 0
        jump_type = 'random'
        for jt, prob in JUMP_TYPE_WEIGHTS.items():
            cumsum += prob
            if r < cumsum:
                jump_type = jt
                break

        system_map = self._piece_system_map.get(piece_id, {})
        page_map = self._piece_page_map.get(piece_id, {})

        dest_indices = None

        if jump_type == 'repeat':
            # Jump backward — weighted: 70% chance of 1-2 systems, 30% chance of 3-4
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
            # Jump forward 1-3 systems
            offsets = [1, 2, 3]
            random.shuffle(offsets)
            for offset in offsets:
                target_sys = src_sys + offset
                if (src_page, target_sys) in system_map:
                    dest_indices = system_map[(src_page, target_sys)]
                    break

        elif jump_type == 'restart':
            # Jump to first system on current page
            if src_sys != 0 and (src_page, 0) in system_map:
                dest_indices = system_map[(src_page, 0)]

        elif jump_type == 'page_jump':
            # Jump to a different page entirely
            other_pages = [p for p in page_map if p != src_page]
            if other_pages:
                dest_page = random.choice(other_pages)
                dest_indices = page_map[dest_page]

        elif jump_type == 'bar_correction':
            # Different bar in same system, or adjacent system
            same_sys = system_map.get((src_page, src_sys), [])
            diff_bar = [i for i in same_sys
                        if self.sequences[i].get('gt_bar_in_system_idx', 0) != src_bar]
            if diff_bar:
                dest_indices = diff_bar
            else:
                # Adjacent system (+-1)
                for offset in [1, -1]:
                    adj = (src_page, src_sys + offset)
                    if adj in system_map:
                        dest_indices = system_map[adj]
                        break

        # Fallback: random different system (covers 'random' type + failed typed samples)
        if dest_indices is None:
            all_indices = []
            for key, indices in system_map.items():
                if key != (src_page, src_sys):
                    all_indices.extend(indices)
            if all_indices:
                dest_indices = all_indices
            else:
                return None

        # Onset bias: prefer landing on onset frames (note attacks / bar boundaries)
        onset_indices = [i for i in dest_indices if self.sequences[i].get('is_onset', False)]
        if onset_indices:
            return random.choice(onset_indices)

        return random.choice(dest_indices)

    def set_jump_prob(self, prob):
        """Set jump probability (for curriculum scheduling)."""
        self.jump_prob = prob
        if prob > 0 and not hasattr(self, '_piece_indices'):
            self._build_jump_indices()

    def _lookup_bar_page_idx(self, piece_id, page_nr, sys_idx, bar_in_sys_idx):
        """Map a page-local (system, bar-in-system) pair to page-local bar index."""
        if self.page_metadata is None:
            return -1

        piece_pages = self.page_metadata.get(piece_id, {})
        page_meta = piece_pages.get(page_nr, {})
        bars_per_system = page_meta.get('bars_per_system', [])
        if 0 <= sys_idx < len(bars_per_system):
            bars_in_sys = bars_per_system[sys_idx]
            if 0 <= bar_in_sys_idx < len(bars_in_sys):
                return int(bars_in_sys[bar_in_sys_idx])
        return -1

    def __getitem__(self, item):

        seq = self.sequences[item]

        # On-the-fly jump augmentation: swap destination while keeping biased prev_gt
        jump_silence_frames = 0
        if (self.jump_prob > 0 and self.augment and
            random.random() < self.jump_prob):
            dest_idx = self._sample_jump_destination(item)
            if dest_idx is not None:
                orig_seq = seq
                seq = dict(self.sequences[dest_idx])  # copy to avoid mutation

                # Break-aligned jump sample: pause first, then a short relocalization window.
                jump_silence_frames = random.randint(
                    JUMP_SILENCE_MIN_FRAMES, JUMP_SILENCE_MAX_FRAMES
                )
                post_silence_frames = random.randint(
                    JUMP_POST_SILENCE_MIN_FRAMES, JUMP_POST_SILENCE_MAX_FRAMES
                )
                seq['start_frame'] = max(
                    0, seq['frame'] - (jump_silence_frames + post_silence_frames)
                )

                # Same-page jumps freeze priors at the source's current position.
                # Cross-page jumps still reset because indices are page-local.
                src_page = int(orig_seq['true_position'][-1])
                dst_page = int(seq['true_position'][-1])
                if src_page == dst_page:
                    src_sys = int(orig_seq.get('gt_system_page_idx', -1))
                    src_bar_in_sys = int(orig_seq.get('gt_bar_in_system_idx', -1))
                    src_bar_page = self._lookup_bar_page_idx(
                        orig_seq['piece_id'], src_page, src_sys, src_bar_in_sys
                    )
                    seq['prev_gt_system_page_idx'] = (
                        src_sys if src_sys >= 0 else orig_seq.get('prev_gt_system_page_idx', -1)
                    )
                    seq['prev_gt_bar_page_idx'] = (
                        src_bar_page if src_bar_page >= 0 else orig_seq.get('prev_gt_bar_page_idx', -1)
                    )
                else:
                    seq['prev_gt_system_page_idx'] = -1
                    seq['prev_gt_bar_page_idx'] = -1

        piece_id = seq['piece_id']
        score = self.scores[piece_id]

        is_onset = seq['is_onset']

        signal = self.performances[piece_id]

        # if signal is provided as a path it should be loaded from the disk
        if isinstance(signal, str):
            signal = load_wav(signal, SAMPLE_RATE)

        start_frame = int(seq['start_frame'])
        frame = int(seq['frame'])
        scale_factor = seq['scale_factor']

        # Cold-start regularization: randomly reset audio context
        # This simulates starting from mid-piece with zero hidden state,
        # forcing the model to rely on audio-score matching rather than
        # trajectory extrapolation. Acts as regularization against overfitting.
        if (self.cold_start_prob > 0 and
            frame >= self.cold_start_min_frames and
            random.random() < self.cold_start_prob):
            # Cold start: use random-length recent audio context
            # Sample context length uniformly between min and max
            max_possible_context = min(self.cold_start_max_context, frame)
            min_context = min(self.cold_start_min_context, max_possible_context)
            recent_context = random.randint(min_context, max_possible_context)
            start_frame = max(0, frame - recent_context)

        start_t = int(start_frame * self.hop_length)
        t = self.frame_size + int(frame * self.hop_length)

        truncated_signal = signal[start_t:t]

        # Insert silence gap before jump destination audio
        # Simulates the pause when a performer jumps to a new location
        if jump_silence_frames > 0:
            truncated_signal = truncated_signal.copy()  # Don't modify original signal
            silence_samples = min(jump_silence_frames * self.hop_length, len(truncated_signal))
            truncated_signal[:silence_samples] = 0

        true_position, page_nr = seq['true_position'][:2], seq['true_position'][-1]

        notes_available = true_position[0] >= 0
        max_y_shift = seq['max_y_shift']
        max_x_shift = seq['max_x_shift']

        page_nr = int(page_nr)

        s = score[page_nr]

        system = np.asarray(seq['true_system'], dtype=np.float32)
        system /= scale_factor

        bar = np.asarray(seq['true_bar'], dtype=np.float32)
        bar /= scale_factor

        true_pos = np.copy(true_position)

        true_pos = true_pos / scale_factor
        width = self.gt_width / scale_factor
        height = seq['height'] / scale_factor

        if self.augment:

            yshift = random.randint(int(max_y_shift[0]/scale_factor), int(max_y_shift[1]/scale_factor))
            xshift = random.randint(int(max_x_shift[0] / scale_factor), int(max_x_shift[1] / scale_factor))

            true_pos[0] += yshift
            true_pos[1] += xshift

            s = np.roll(s, yshift, 0)
            s = np.roll(s, xshift, 1)

            # System [center_x, center_y, width, height]
            system[0] += xshift
            system[1] += yshift

            bar[0] += xshift
            bar[1] += yshift

            # pad signal randomly by 0-20 frames (0-1seconds)
            truncated_signal = np.pad(truncated_signal, (random.randint(0, int(self.fps)) * self.hop_length, 0),
                                      mode='constant')

        center_y, center_x = true_pos

        target = []

        if self.system_only:
            # System-only mode: output only system box as class 0
            target.append([0, 0, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])
        else:
            # Original behavior: note as class 0, optionally bar/system as class 1/2
            if notes_available:
                target.append([0, 0, center_x/s.shape[1], center_y/s.shape[0], width/s.shape[1], height/s.shape[0]])

            if self.predict_sb:
                target.append([0, 1, bar[0] / s.shape[1], bar[1] / s.shape[0], bar[2] / s.shape[1], bar[3] / s.shape[0]])
                target.append([0, 2, system[0]/s.shape[1], system[1]/s.shape[0], system[2]/s.shape[1], system[3]/s.shape[0]])

        target = np.asarray(target, dtype=np.float32)

        unscaled_targets = np.copy(target)
        unscaled_targets[:, 2] *= s.shape[1]
        unscaled_targets[:, 3] *= s.shape[0]
        unscaled_targets[:, 4] *= s.shape[1]
        unscaled_targets[:, 5] *= s.shape[0]

        unscaled_targets[:, 2:] *= scale_factor

        interpol_c2o = self.interpol_c2o[piece_id][page_nr]
        add_per_staff = [self.staff_coords[piece_id][page_nr], self.add_per_staff[piece_id][page_nr]]
        piece_name = f"{self.piece_names[piece_id]}_page_{page_nr}"

        # only use two minutes of audio to avoid GPU memory issues
        truncated_signal = truncated_signal[- (120 * self.sample_rate):]

        sample = {'performance': truncated_signal,  'score': s[None], 'target': target,
                  'file_name': piece_name, 'is_onset': is_onset, 'interpol_c2o': interpol_c2o,
                  'add_per_staff': add_per_staff, 'scale_factor': scale_factor, 'unscaled_target': unscaled_targets,
                  't': t,
                  '_effective_seq': seq}  # Pass effective seq for selection_getitem (may differ from original after jump)

        if self.transform:
            sample = self.transform(sample)

        return sample


# --- Selection model dataset support ---

class SelectionCustomBatch:
    """Batch collator for selection model. Includes page layout metadata."""

    def __init__(self, batch):
        self.file_names = [x['file_name'] for x in batch]
        self.perf = [torch.as_tensor(x['performance'], dtype=torch.float32) for x in batch]

        self.scores = torch.as_tensor(np.stack([x['score'] for x in batch]), dtype=torch.float32)
        self.scale_factors = torch.FloatTensor([x['scale_factor'] for x in batch]).float().unsqueeze(-1)

        # Selection-specific: variable-length page layout per batch item
        self.system_boxes = [torch.as_tensor(x['system_boxes'], dtype=torch.float32) for x in batch]
        self.bar_boxes = [torch.as_tensor(x['bar_boxes'], dtype=torch.float32) for x in batch]
        self.bars_per_system = [x['bars_per_system'] for x in batch]

        self.gt_system_idx = torch.LongTensor([x['gt_system_page_idx'] for x in batch])
        self.gt_bar_in_sys = torch.LongTensor([x['gt_bar_in_system_idx'] for x in batch])
        self.prev_gt_system_idx = torch.LongTensor([x.get('prev_gt_system_page_idx', -1) for x in batch])
        self.prev_gt_bar_page_idx = torch.LongTensor([x.get('prev_gt_bar_page_idx', -1) for x in batch])

        # gt_valid: all True since invalid samples are filtered at load time.
        # Passed through to loss for defensive masking of edge cases.
        self.gt_valid = torch.ones(len(batch), dtype=torch.bool)

        # GT note position (bar-local, for note regression loss)
        self.gt_note_position = torch.as_tensor(
            np.stack([x['gt_note_bar_local'] for x in batch]), dtype=torch.float32
        )

        # Original targets (for backward compatibility with eval)
        targets = []
        unscaled_targets = []
        for i, x in enumerate(batch):
            if x['target'] is not None:
                target = x['target'].copy()
                target[:, 0] = i
                targets.append(target)
                unscaled_target = x['unscaled_target'].copy()
                unscaled_target[:, 0] = i
                unscaled_targets.append(unscaled_target)

        if targets:
            self.targets = torch.as_tensor(np.concatenate(targets), dtype=torch.float32)
            self.unscaled_targets = torch.as_tensor(np.concatenate(unscaled_targets), dtype=torch.float32)
        else:
            self.targets = torch.zeros(0, 6, dtype=torch.float32)
            self.unscaled_targets = torch.zeros(0, 6, dtype=torch.float32)

        self.interpols = [x['interpol_c2o'] for x in batch]
        self.add_per_staff = [x['add_per_staff'] for x in batch]

    def pin_memory(self):
        self.scores = self.scores.pin_memory()
        self.perf = [p.pin_memory() for p in self.perf]
        self.gt_system_idx = self.gt_system_idx.pin_memory()
        self.gt_bar_in_sys = self.gt_bar_in_sys.pin_memory()
        self.gt_note_position = self.gt_note_position.pin_memory()
        self.prev_gt_system_idx = self.prev_gt_system_idx.pin_memory()
        self.prev_gt_bar_page_idx = self.prev_gt_bar_page_idx.pin_memory()
        self.gt_valid = self.gt_valid.pin_memory()
        self.targets = self.targets.pin_memory()
        self.unscaled_targets = self.unscaled_targets.pin_memory()
        self.scale_factors = self.scale_factors.pin_memory()
        return self


def selection_collate_wrapper(batch):
    return SelectionCustomBatch(batch)


def selection_getitem(dataset, item):
    """
    Extended __getitem__ for selection model.
    Wraps the standard SequenceDataset sample with page layout metadata.

    Uses _effective_seq from the sample dict (set by __getitem__'s jump augmentation)
    instead of reading dataset.sequences[item] directly. This ensures that when a
    jump swap occurs, the GT metadata matches the swapped destination.
    """
    # Get standard sample first (handles augmentation, audio, jump swap)
    sample = dataset.__getitem__(item)

    # Pop effective seq from sample dict (may differ from original after jump augmentation)
    seq = sample.pop('_effective_seq', dataset.sequences[item])

    piece_id = seq['piece_id']
    page_nr = int(seq['true_position'][-1])
    scale_factor = seq['scale_factor']

    # Look up page metadata
    pm = dataset.page_metadata[piece_id][page_nr]
    sys_boxes = pm['system_boxes'].copy()  # [N_sys, 4] xywh in padded coords
    bar_boxes = pm['bar_boxes'].copy()     # [N_bar, 4] xywh in padded coords

    # Scale to model input space
    sys_boxes /= scale_factor
    bar_boxes /= scale_factor

    # Apply same augmentation shift as the score image
    if dataset.augment:
        # Recover the shift that was applied — it's stored in the difference
        # between the original and augmented GT positions
        original_system = np.asarray(seq['true_system'], dtype=np.float32) / scale_factor
        augmented_system = sample['target'][sample['target'][:, 1] == 2]  # class 2 = system
        if len(augmented_system) > 0:
            aug_cx = augmented_system[0, 2] * sample['score'].shape[2]
            aug_cy = augmented_system[0, 3] * sample['score'].shape[1]
            xshift = aug_cx - original_system[0]
            yshift = aug_cy - original_system[1]
        else:
            xshift, yshift = 0.0, 0.0

        sys_boxes[:, 0] += xshift
        sys_boxes[:, 1] += yshift
        bar_boxes[:, 0] += xshift
        bar_boxes[:, 1] += yshift

    sample['system_boxes'] = sys_boxes
    sample['bar_boxes'] = bar_boxes
    sample['bars_per_system'] = pm['bars_per_system']
    sample['gt_system_page_idx'] = seq.get('gt_system_page_idx', 0)
    sample['gt_bar_in_system_idx'] = seq.get('gt_bar_in_system_idx', 0)
    sample['prev_gt_system_page_idx'] = seq.get('prev_gt_system_page_idx', -1)
    sample['prev_gt_bar_page_idx'] = seq.get('prev_gt_bar_page_idx', -1)

    # Compute GT note position in bar-local coords [0, 1]
    gt_bar_global_in_page = 0
    bps = pm['bars_per_system']
    gt_sys_idx = sample['gt_system_page_idx']
    gt_bar_in_sys = sample['gt_bar_in_system_idx']
    if gt_sys_idx < len(bps) and gt_bar_in_sys < len(bps[gt_sys_idx]):
        gt_bar_global_in_page = bps[gt_sys_idx][gt_bar_in_sys]

    if gt_bar_global_in_page < len(bar_boxes):
        bar = bar_boxes[gt_bar_global_in_page]  # xywh
        bar_x1 = bar[0] - bar[2] / 2
        bar_y1 = bar[1] - bar[3] / 2

        # GT note position in model input space
        true_pos = seq['true_position'][:2].astype(np.float32) / scale_factor
        if dataset.augment:
            true_pos[1] += xshift  # cx
            true_pos[0] += yshift  # cy

        note_local_cx = (true_pos[1] - bar_x1) / max(bar[2], 1e-6)
        note_local_cy = (true_pos[0] - bar_y1) / max(bar[3], 1e-6)
        sample['gt_note_bar_local'] = np.clip(np.array([note_local_cx, note_local_cy], dtype=np.float32), 0, 1)
    else:
        sample['gt_note_bar_local'] = np.array([0.5, 0.5], dtype=np.float32)

    return sample


def load_dataset(paths, augment=False, scale_width=416, split_files=None, ir_path=None,
                 only_onsets=False, load_audio=True, predict_sb=False, system_only=False,
                 cold_start_prob=0.0, cold_start_min_frames=50,
                 cold_start_min_context=20, cold_start_max_context=400,
                 jump_prob=0.0):

    scores = {}
    piece_names = {}
    all_sequences = []
    total_invalid = 0
    performances = {}
    interpol_c2os = {}
    staff_coords_all = {}
    add_per_staff_all = {}
    params = []

    files = []
    if split_files is not None:
        assert len(split_files) == len(paths)

        for idx, split_file in enumerate(split_files):
            split = load_yaml(split_file)
            files.extend([os.path.join(paths[idx], f'{file}.npz') for file in split['files']])

    else:
        for path in paths:
            files.extend(glob.glob(os.path.join(path, '*.npz')))

    # Phase 1: Identify audio sources for all files to enable score/audio deduplication
    # This dramatically reduces memory for jump-augmented datasets
    file_audio_sources = {}  # file_idx -> (audio_source_name, audio_source_dir)
    audio_source_to_first_idx = {}  # (audio_source_name, dir) -> first file_idx that loaded it

    print(f'Scanning {len(files)} file(s) for audio sources...')
    for i, score_path in enumerate(files):
        piece_name = os.path.basename(score_path)[:-4]
        piece_dir = os.path.dirname(score_path)

        # Quick scan to check for audio_source field
        try:
            npz = np.load(score_path, allow_pickle=True)
            if 'audio_source' in npz:
                audio_source = str(npz['audio_source'].item())
                # Look for the audio source in known paths (same dir, or train dir for jump dirs)
                possible_dirs = [piece_dir]
                # Check parent directory patterns (e.g., msmd_train_jump -> msmd_train)
                if '_jump' in piece_dir:
                    base_dir = piece_dir.replace('_jump', '')
                    if os.path.exists(base_dir):
                        possible_dirs.append(base_dir)

                audio_dir = None
                for d in possible_dirs:
                    if os.path.exists(os.path.join(d, f'{audio_source}.wav')):
                        audio_dir = d
                        break

                if audio_dir is None:
                    audio_dir = piece_dir  # Fallback

                file_audio_sources[i] = (audio_source, audio_dir)
            else:
                file_audio_sources[i] = (piece_name, piece_dir)
            npz.close()
        except Exception as e:
            print(f"Warning: Could not scan {score_path}: {e}")
            file_audio_sources[i] = (piece_name, piece_dir)

    # Build mapping of unique audio sources
    for i, (audio_source, audio_dir) in file_audio_sources.items():
        key = (audio_source, audio_dir)
        if key not in audio_source_to_first_idx:
            audio_source_to_first_idx[key] = i

    unique_sources = len(audio_source_to_first_idx)
    print(f'Found {unique_sources} unique audio sources for {len(files)} files (deduplication ratio: {len(files)/max(unique_sources,1):.1f}x)')

    for i, score_path in enumerate(files):
        piece_name = os.path.basename(score_path)[:-4]
        piece_dir = os.path.dirname(score_path)
        audio_source, audio_dir = file_audio_sources[i]
        source_key = (audio_source, audio_dir)

        # Check if this piece should load its own scores or reuse from another
        is_primary = (audio_source_to_first_idx[source_key] == i)

        params.append(dict(
            i=i,
            piece_name=piece_name,
            path=piece_dir,
            scale_width=scale_width,
            load_audio=load_audio,
            is_primary=is_primary,  # Only primary loads scores/audio
            source_idx=audio_source_to_first_idx[source_key],  # Which idx to get scores from
        ))

    print(f'Loading {len(params)} file(s)...')

    # Use spawn instead of fork to avoid deadlocks with CUDA/audio libs
    # Fork can hang when CUDA, librosa, or OpenMP have been initialized
    mp_start = os.getenv("CODA_DATASET_START_METHOD", "spawn")
    mp_workers = int(os.getenv("CODA_DATASET_WORKERS", "8"))

    if mp_workers <= 0:
        # Sequential loading (safest, use if parallel still hangs)
        results = [load_sequences(p) for p in tqdm(params)]
    else:
        with get_context(mp_start).Pool(mp_workers) as pool:
            results = list(tqdm(pool.imap_unordered(load_sequences, params), total=len(params)))

    # Phase 2: Process results with score deduplication
    # Non-primary files return None for scores/audio/interpol (lightweight mode)
    page_metadata_all = {}
    for result in results:
        i, score, signals, piece_name, sequences, interpol_c2o, staff_coords, add_per_staff, page_metadata = result

        source_idx = params[i]['source_idx']
        is_primary = params[i]['is_primary']

        if is_primary:
            # Primary piece - store scores/audio/interpol normally
            scores[i] = score
            performances[i] = signals
            interpol_c2os[i] = interpol_c2o
            staff_coords_all[i] = staff_coords
            add_per_staff_all[i] = add_per_staff
            if page_metadata is not None:
                page_metadata_all[i] = page_metadata
        # Non-primary: scores/audio/interpol are None, will resolve in phase 3

        piece_names[i] = piece_name

        # Sequences already have piece_id pointing to source_idx (set in load_sequences)
        # Filter by onset and gt_valid — invalid GT samples excluded at load time
        n_before = len(all_sequences)
        for seq in sequences:
            if only_onsets and not seq['is_onset']:
                continue
            if not seq.get('gt_valid', True):
                continue
            all_sequences.append(seq)
        n_added = len(all_sequences) - n_before
        n_invalid = len(sequences) - n_added
        if n_invalid > 0:
            total_invalid += n_invalid

    if total_invalid > 0:
        print(f'[Dataset] Filtered {total_invalid} samples with invalid GT labels '
              f'({total_invalid}/{total_invalid + len(all_sequences)} = '
              f'{total_invalid/(total_invalid + len(all_sequences))*100:.2f}%)')

    # Phase 3: Resolve deduplication - point variants to source data
    # This shares references (no memory copy!)
    for i in range(len(params)):
        source_idx = params[i]['source_idx']
        if source_idx != i:
            scores[i] = scores[source_idx]
            performances[i] = performances[source_idx]
            interpol_c2os[i] = interpol_c2os[source_idx]
            staff_coords_all[i] = staff_coords_all[source_idx]
            add_per_staff_all[i] = add_per_staff_all[source_idx]
            if source_idx in page_metadata_all:
                page_metadata_all[i] = page_metadata_all[source_idx]

    print('Done loading.')

    if ir_path is not None:
        print('Using Impulse Response Augmentation')
        ir_aug = ImpulseResponse(ir_paths=ir_path, ir_prob=0.5)
        transform = torchvision.transforms.Compose([ir_aug])
    else:
        transform = None

    return SequenceDataset(scores, performances, all_sequences, piece_names, interpol_c2os, staff_coords_all,
                           add_per_staff_all, augment=augment, transform=transform, predict_sb=predict_sb,
                           system_only=system_only, cold_start_prob=cold_start_prob,
                           cold_start_min_frames=cold_start_min_frames,
                           cold_start_min_context=cold_start_min_context,
                           cold_start_max_context=cold_start_max_context,
                           page_metadata=page_metadata_all if page_metadata_all else None,
                           jump_prob=jump_prob)
