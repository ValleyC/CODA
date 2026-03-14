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


def _compute_ce_loss(logits_flat, counts, gt_indices, gt_valid, B,
                     label_smoothing=0.0, debug_assert=False):
    """
    Compute per-sample cross-entropy over variable-length candidates.

    Returns:
        loss: scalar tensor (mean over valid samples)
        n_correct: int
        n_valid: int
    """
    device = gt_indices.device
    loss = torch.tensor(0.0, device=device)
    n_correct = 0
    n_valid = 0
    offset = 0

    for b in range(B):
        n = counts[b]
        if n > 0:
            if gt_valid is not None and not gt_valid[b]:
                offset += n
                continue

            target_val = gt_indices[b].item()
            if debug_assert:
                assert 0 <= target_val < n, \
                    f"Target {target_val} out of bounds [0, {n}) for batch item {b}"
            if target_val < 0 or target_val >= n:
                offset += n
                continue

            logits_b = logits_flat[offset:offset + n].unsqueeze(0)  # [1, n]
            target_b = gt_indices[b:b + 1]                          # [1]

            if label_smoothing > 0 and n > 1:
                # Manual label smoothing for variable-length CE
                log_probs = F.log_softmax(logits_b, dim=1)
                smooth = label_smoothing / n
                targets_smooth = torch.full_like(log_probs, smooth)
                targets_smooth[0, target_val] = 1.0 - label_smoothing + smooth
                loss = loss + (-targets_smooth * log_probs).sum(dim=1).mean()
            else:
                loss = loss + F.cross_entropy(logits_b, target_b)

            pred_b = logits_b.argmax(dim=1)
            n_correct += (pred_b == target_b).sum().item()
            n_valid += 1
        offset += n

    loss = loss / max(n_valid, 1)
    return loss, n_correct, n_valid


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
    device = gt_system_idx.device
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
    if note_positions.shape[0] > 0:
        note_mask = bar_valid if bar_valid is not None else gt_valid
        if note_mask is not None:
            valid_mask = note_mask.bool()
            if valid_mask.any():
                note_loss = F.mse_loss(note_positions[valid_mask], gt_note_position[valid_mask])
            else:
                note_loss = torch.tensor(0.0, device=device)
        else:
            note_loss = F.mse_loss(note_positions, gt_note_position)
    else:
        note_loss = torch.tensor(0.0, device=device)

    # Static weighting
    loss = system_weight * sys_loss + bar_weight * bar_loss + note_weight * note_loss

    return {
        'loss': loss,
        'sys_loss': sys_loss,
        'bar_loss': bar_loss,
        'note_loss': note_loss,
        'sys_acc': sys_correct / max(n_valid, 1),
        'bar_acc': bar_correct / max(n_valid_bar, 1),
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
        device = gt_system_idx.device
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
        if note_positions.shape[0] > 0:
            note_mask = bar_valid if bar_valid is not None else gt_valid
            if note_mask is not None:
                valid_mask = note_mask.bool()
                if valid_mask.any():
                    note_loss = F.mse_loss(note_positions[valid_mask], gt_note_position[valid_mask])
                else:
                    note_loss = torch.tensor(0.0, device=device)
            else:
                note_loss = F.mse_loss(note_positions, gt_note_position)
        else:
            note_loss = torch.tensor(0.0, device=device)

        # Uncertainty weighting: L_i / (2 * exp(s_i)) + s_i / 2
        precision_sys = torch.exp(-self.log_var_sys)
        precision_bar = torch.exp(-self.log_var_bar)
        precision_note = torch.exp(-self.log_var_note)

        loss = (0.5 * precision_sys * sys_loss + 0.5 * self.log_var_sys
              + 0.5 * precision_bar * bar_loss + 0.5 * self.log_var_bar
              + 0.5 * precision_note * note_loss + 0.5 * self.log_var_note)

        # Effective weights for logging (higher precision = higher weight)
        w_sys = precision_sys.item()
        w_bar = precision_bar.item()
        w_note = precision_note.item()

        return {
            'loss': loss,
            'sys_loss': sys_loss,
            'bar_loss': bar_loss,
            'note_loss': note_loss,
            'sys_acc': sys_correct / max(n_valid, 1),
            'bar_acc': bar_correct / max(n_valid_bar, 1),
            'w_sys': w_sys,
            'w_bar': w_bar,
            'w_note': w_note,
        }
