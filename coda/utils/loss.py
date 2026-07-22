"""
Loss functions for selection-based hierarchical score following.

Three components:
  1. System selection: cross-entropy over page systems
  2. Bar selection: cross-entropy over bars in selected system
  3. Note regression: MSE on bar-local sigmoid position

Supports:
  - Static weighting (system_weight, bar_weight, note_weight)
  - Uncertainty weighting (Kendall et al., "Multi-Task Learning Using
    Uncertainty to Weigh Losses") via learnable log-variance per task
  - Label smoothing for system/bar CE losses
  - Separate validity masks for system vs bar/note (for scheduled sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _compute_ce_loss(logits_flat, counts, gt_indices, gt_valid, B,
                     label_smoothing=0.0, debug_assert=False):
    """
    Compute per-sample cross-entropy over variable-length candidates.

    Returns:
        loss: scalar tensor (mean over valid samples)
        n_correct: scalar tensor
        n_valid: scalar tensor
    """
    if B != len(counts) or gt_indices.shape[0] != B:
        raise ValueError("counts and gt_indices must have one entry per batch item")

    device = logits_flat.device
    counts_tensor = torch.as_tensor(counts, device=device, dtype=torch.long)
    max_count = max(counts, default=0)
    if max_count == 0:
        zero = logits_flat.sum() * 0.0
        return zero, zero.detach(), torch.zeros((), device=device, dtype=torch.long)

    # One padded CE kernel replaces B small kernels and dozens of .item()
    # synchronizations. CopySlices/pad_sequence preserve gradients back to the
    # original flat candidate logits.
    segments = torch.split(logits_flat, counts)
    padded_logits = pad_sequence(
        segments, batch_first=True, padding_value=float('-inf')
    )
    candidate_mask = (
        torch.arange(max_count, device=device).unsqueeze(0)
        < counts_tensor.unsqueeze(1)
    )

    has_candidates = counts_tensor > 0
    target_in_bounds = (gt_indices >= 0) & (gt_indices < counts_tensor)
    eligible = has_candidates
    if gt_valid is not None:
        eligible = eligible & gt_valid.bool()
    valid = eligible & target_in_bounds

    invalid_targets = eligible & ~target_in_bounds
    if debug_assert and bool(invalid_targets.any()):
        invalid_rows = invalid_targets.nonzero(as_tuple=False).flatten().cpu().tolist()
        raise AssertionError(f"Invalid variable-candidate targets at batch rows {invalid_rows}")

    # Make empty/invalid rows numerically safe even though their losses are
    # masked. In particular, avoid a log_softmax row containing only -inf.
    empty_rows = counts_tensor == 0
    if 0 in counts:
        padded_logits = padded_logits.clone()
        padded_logits[empty_rows, 0] = 0.0
    safe_targets = torch.where(valid, gt_indices, torch.zeros_like(gt_indices))

    if label_smoothing > 0:
        log_probs = F.log_softmax(padded_logits, dim=1)
        nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.masked_fill(~candidate_mask, 0.0).sum(dim=1)
        smooth = smooth / counts_tensor.clamp_min(1)
        per_sample_loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_sample_loss = F.cross_entropy(
            padded_logits, safe_targets, reduction='none'
        )

    per_sample_loss = torch.where(valid, per_sample_loss, torch.zeros_like(per_sample_loss))
    n_valid = valid.sum()
    loss = per_sample_loss.sum() / n_valid.clamp_min(1)
    n_correct = ((padded_logits.argmax(dim=1) == gt_indices) & valid).sum()
    return loss, n_correct, n_valid


def _masked_note_mse(note_positions, gt_note_position, valid_mask):
    """Mean note regression loss without a GPU-synchronizing ``any()``."""
    if note_positions.shape[0] == 0:
        return note_positions.sum() * 0.0
    per_sample = (note_positions - gt_note_position).pow(2).mean(dim=1)
    if valid_mask is None:
        return per_sample.mean()
    weights = valid_mask.to(dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def selection_loss(output, gt_system_idx, gt_bar_in_sys, gt_note_position,
                   system_weight=1.0, bar_weight=1.0, note_weight=1.0,
                   gt_valid=None, bar_note_valid=None,
                   debug_assert=False, label_smoothing=0.0):
    """
    Compute selection losses with static weighting.

    Args:
        output: dict from SelectionCascadeModel.forward()
        gt_system_idx: [B] LongTensor -- GT system page-local index
        gt_bar_in_sys: [B] LongTensor -- GT bar index within selected system
        gt_note_position: [B, 2] FloatTensor -- GT note (cx, cy) in bar-local [0,1]
        system_weight: weight for system CE loss
        bar_weight: weight for bar CE loss
        note_weight: weight for note MSE loss
        gt_valid: [B] BoolTensor -- validity mask for ALL tasks.
        bar_note_valid: [B] BoolTensor -- additional mask for bar/note only
                        (used by scheduled sampling when GT bar not in predicted system).
                        If None, uses gt_valid for all tasks.
        debug_assert: If True, raise on out-of-bounds targets (dev mode).
        label_smoothing: Label smoothing factor for CE losses (0.0 = none).

    Returns:
        loss_dict with 'loss', 'sys_loss', 'bar_loss', 'note_loss',
        'sys_acc', 'bar_acc'
    """
    B = gt_system_idx.shape[0]

    # System CE -- always uses gt_valid (system targets are always correct)
    sys_loss, sys_correct, n_valid = _compute_ce_loss(
        output['sys_logits'], output['sys_counts'], gt_system_idx,
        gt_valid, B, label_smoothing=label_smoothing, debug_assert=debug_assert,
    )

    # Bar CE -- uses bar_note_valid (may mask when predicted system != GT system)
    bar_valid = bar_note_valid if bar_note_valid is not None else gt_valid
    bar_loss, bar_correct, n_valid_bar = _compute_ce_loss(
        output['bar_logits'], output['bar_counts'], gt_bar_in_sys,
        bar_valid, B, label_smoothing=label_smoothing, debug_assert=debug_assert,
    )

    # Note MSE -- uses bar_note_valid
    note_positions = output['note_positions']
    note_mask = bar_valid if bar_valid is not None else gt_valid
    note_loss = _masked_note_mse(note_positions, gt_note_position, note_mask)

    # Static weighting
    loss = system_weight * sys_loss + bar_weight * bar_loss + note_weight * note_loss

    return {
        'loss': loss,
        'sys_loss': sys_loss,
        'bar_loss': bar_loss,
        'note_loss': note_loss,
        'sys_acc': sys_correct.float() / n_valid.clamp_min(1),
        'bar_acc': bar_correct.float() / n_valid_bar.clamp_min(1),
    }


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty weighting for multi-task loss (Kendall et al., 2018).

    Each task has a learnable log-variance parameter s_i.
    Loss_i is weighted as: L_i / (2 * exp(s_i)) + s_i / 2

    When a task is noisy (high variance), exp(s_i) grows -> weight decreases.
    When a task is clean (low variance), exp(s_i) shrinks -> weight increases.
    The s_i/2 regularizer prevents all weights from going to infinity.
    """

    def __init__(self, label_smoothing=0.0, debug_assert=False):
        super().__init__()
        # Learnable log-variance per task (initialized to 0 -> equal weighting)
        self.log_var_sys = nn.Parameter(torch.zeros(1))
        self.log_var_bar = nn.Parameter(torch.zeros(1))
        self.log_var_note = nn.Parameter(torch.zeros(1))
        self.label_smoothing = label_smoothing
        self.debug_assert = debug_assert

    def forward(self, output, gt_system_idx, gt_bar_in_sys, gt_note_position,
                gt_valid=None, bar_note_valid=None, **kwargs):
        """
        Compute uncertainty-weighted selection loss.

        Ignores system_weight/bar_weight/note_weight kwargs -- weights are learned.
        """
        B = gt_system_idx.shape[0]

        # System CE -- always uses gt_valid
        sys_loss, sys_correct, n_valid = _compute_ce_loss(
            output['sys_logits'], output['sys_counts'], gt_system_idx,
            gt_valid, B, label_smoothing=self.label_smoothing,
            debug_assert=self.debug_assert,
        )

        # Bar CE -- uses bar_note_valid
        bar_valid = bar_note_valid if bar_note_valid is not None else gt_valid
        bar_loss, bar_correct, n_valid_bar = _compute_ce_loss(
            output['bar_logits'], output['bar_counts'], gt_bar_in_sys,
            bar_valid, B, label_smoothing=self.label_smoothing,
            debug_assert=self.debug_assert,
        )

        # Note MSE -- uses bar_note_valid
        note_positions = output['note_positions']
        note_mask = bar_valid if bar_valid is not None else gt_valid
        note_loss = _masked_note_mse(note_positions, gt_note_position, note_mask)

        # Uncertainty weighting: L_i / (2 * exp(s_i)) + s_i / 2
        precision_sys = torch.exp(-self.log_var_sys)
        precision_bar = torch.exp(-self.log_var_bar)
        precision_note = torch.exp(-self.log_var_note)

        loss = (0.5 * precision_sys * sys_loss + 0.5 * self.log_var_sys
              + 0.5 * precision_bar * bar_loss + 0.5 * self.log_var_bar
              + 0.5 * precision_note * note_loss + 0.5 * self.log_var_note)

        return {
            'loss': loss,
            'sys_loss': sys_loss,
            'bar_loss': bar_loss,
            'note_loss': note_loss,
            'sys_acc': sys_correct.float() / n_valid.clamp_min(1),
            'bar_acc': bar_correct.float() / n_valid_bar.clamp_min(1),
            'w_sys': precision_sys.detach(),
            'w_bar': precision_bar.detach(),
            'w_note': precision_note.detach(),
        }
