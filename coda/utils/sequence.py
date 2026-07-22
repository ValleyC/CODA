"""Utilities for correctly handling variable-length temporal sequences."""

from typing import Optional, Tuple

import torch


def take_last_valid_window(
    sequence: torch.Tensor,
    lengths: Optional[torch.Tensor],
    window: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Take each sample's last valid frames from a right-padded batch.

    The previous implementation sliced the batch-wide tensor from the right,
    which selects padding for shorter samples. This function gathers relative
    to each sample's own valid length and returns a left-aligned padded window.
    """
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, time, features]")
    if window is None or sequence.shape[1] <= window:
        return sequence, lengths
    if window <= 0:
        raise ValueError("window must be positive")

    if lengths is None:
        return sequence[:, -window:, :], None

    batch, time, features = sequence.shape
    lengths = lengths.to(device=sequence.device, dtype=torch.long).clamp(max=time)
    if torch.any(lengths <= 0):
        raise ValueError("all sequence lengths must be positive")

    window_lengths = lengths.clamp(max=window)
    positions = torch.arange(window, device=sequence.device).unsqueeze(0)
    starts = (lengths - window_lengths).unsqueeze(1)
    gather_indices = (starts + positions).clamp(max=time - 1)
    gather_indices = gather_indices.unsqueeze(-1).expand(batch, window, features)
    gathered = torch.gather(sequence, dim=1, index=gather_indices)

    valid = positions < window_lengths.unsqueeze(1)
    gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
    return gathered, window_lengths
