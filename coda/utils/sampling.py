"""Restart-stable sampling primitives for stochastic training datasets."""

from typing import Iterator, Sized, Tuple

import torch
from torch.utils.data import Sampler

from coda.utils.general import stable_named_seed


def epoch_sample_seed(base_seed: int, epoch: int, index: int) -> int:
    """Derive the local augmentation seed for one sample in one epoch."""
    return stable_named_seed(
        base_seed, f"training-sample:{int(epoch)}:{int(index)}"
    )


def training_phase_seed(base_seed: int, phase_tag: str) -> int:
    """Namespace a deterministic data stream by curriculum phase."""
    return stable_named_seed(base_seed, f"training-phase:{phase_tag}")


class EpochSeededRandomSampler(Sampler):
    """Yield a deterministic permutation tagged with its epoch.

    The epoch tag travels through ``BatchSampler`` to the dataset wrapper, so
    per-sample augmentation can be derived from ``(seed, epoch, index)`` rather
    than from uncheckpointed persistent-worker RNG state.
    """

    def __init__(self, data_source: Sized, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(
            stable_named_seed(self.seed, f"training-epoch:{self.epoch}")
        )
        order = torch.randperm(len(self), generator=generator).tolist()
        return iter((self.epoch, index) for index in order)
