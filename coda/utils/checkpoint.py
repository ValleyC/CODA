"""Atomic, resumable training checkpoints."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def atomic_torch_save(payload: Any, path: str) -> None:
    """Write a torch artifact atomically so interruptions cannot corrupt it."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        # ``os.replace`` prevents readers from observing a partial file, but
        # does not by itself guarantee that the bytes reached stable storage.
        # Flush the artifact before publication, then flush the containing
        # directory entry on POSIX systems so a reported checkpoint survives
        # a power or host failure.
        # Windows' CRT requires a writable descriptor for ``fsync`` even
        # though no further writes occur; ``r+b`` works portably on both OSes.
        with open(temporary_path, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if os.name == "posix":
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def extract_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Accept either a model-only state dict or a structured checkpoint."""
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    if any(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    return state


def validate_resume_checkpoint(checkpoint: Dict[str, Any], path: str = "checkpoint") -> None:
    """Reject artifacts that cannot represent an exact training continuation."""
    if "model_state_dict" not in checkpoint:
        raise ValueError("--resume_state requires a structured training checkpoint")
    if checkpoint.get("partial_epoch"):
        recovery_path = checkpoint.get("resume_from_completed_checkpoint")
        message = (
            f"{path} is a diagnostic partial-epoch checkpoint. Resuming it "
            "from batch zero would duplicate optimizer updates from the "
            "already-trained epoch prefix."
        )
        if recovery_path:
            message += f" Resume the last completed checkpoint instead: {recovery_path}"
        raise ValueError(message)


def make_training_checkpoint(
    model: nn.Module,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    min_loss: float,
    best_streaming_bar_acc: float,
    best_epoch: int,
    args: Optional[dict] = None,
    metrics: Optional[dict] = None,
) -> Dict[str, Any]:
    """Capture all state required for an exact next-epoch resume."""
    checkpoint = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "criterion_state_dict": criterion.state_dict() if isinstance(criterion, nn.Module) else None,
        "min_loss": float(min_loss),
        "best_streaming_bar_acc": float(best_streaming_bar_acc),
        "best_epoch": int(best_epoch),
        "args": args or {},
        "metrics": metrics or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    return checkpoint


def restore_rng_state(checkpoint: Dict[str, Any]) -> None:
    """Restore RNG streams when resuming a structured checkpoint."""
    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
