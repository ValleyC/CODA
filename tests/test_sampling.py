import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from coda.utils.general import isolated_python_numpy_torch_seed, stable_named_seed
from coda.utils.sampling import (
    EpochSeededRandomSampler,
    epoch_sample_seed,
    training_phase_seed,
)


class _StochasticEpochDataset(Dataset):
    def __init__(self, size, seed):
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, tagged_index):
        epoch, index = tagged_index
        seed = epoch_sample_seed(self.seed, epoch, index)
        with isolated_python_numpy_torch_seed(seed):
            return torch.tensor([
                epoch,
                index,
                random.random(),
                np.random.random(),
                torch.rand(()).item(),
            ], dtype=torch.float64)


def _make_loader(epoch):
    dataset = _StochasticEpochDataset(41, seed=4711)
    sampler = EpochSeededRandomSampler(dataset, seed=4711)
    sampler.set_epoch(epoch)
    generator = torch.Generator().manual_seed(
        stable_named_seed(4711, "test-loader-workers")
    )
    loader = DataLoader(
        dataset,
        batch_size=7,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context='spawn',
        generator=generator,
    )
    return loader, sampler


def _collect(loader):
    return torch.cat(list(loader), dim=0)


def _shutdown(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()


class EpochSeededRandomSamplerTests(unittest.TestCase):
    def test_training_phases_are_distinct_but_stable(self):
        phase_one = training_phase_seed(4711, 'coda_full_phase1')
        self.assertEqual(
            phase_one, training_phase_seed(4711, 'coda_full_phase1')
        )
        self.assertNotEqual(
            phase_one, training_phase_seed(4711, 'coda_full_phase2')
        )

    def test_epoch_permutation_is_restart_stable(self):
        data = list(range(97))
        first = EpochSeededRandomSampler(data, seed=4711)
        resumed = EpochSeededRandomSampler(data, seed=4711)
        first.set_epoch(8)
        resumed.set_epoch(8)

        first_order = list(first)
        resumed_order = list(resumed)
        self.assertEqual(first_order, resumed_order)
        self.assertEqual({epoch for epoch, _ in first_order}, {8})
        self.assertEqual(sorted(index for _, index in first_order), data)

    def test_epoch_and_run_seed_change_the_permutation(self):
        data = list(range(31))
        sampler = EpochSeededRandomSampler(data, seed=10)
        sampler.set_epoch(2)
        epoch_two = list(sampler)
        sampler.set_epoch(3)
        epoch_three = list(sampler)

        other_run = EpochSeededRandomSampler(data, seed=11)
        other_run.set_epoch(2)
        self.assertNotEqual(epoch_two, epoch_three)
        self.assertNotEqual(epoch_two, list(other_run))

    def test_persistent_workers_match_fresh_epoch_restart(self):
        persistent, persistent_sampler = _make_loader(epoch=4)
        restarted = None
        try:
            _collect(persistent)
            persistent_sampler.set_epoch(5)
            continued_epoch = _collect(persistent)

            restarted, _ = _make_loader(epoch=5)
            restarted_epoch = _collect(restarted)
            torch.testing.assert_close(
                continued_epoch, restarted_epoch, rtol=0, atol=0
            )
        finally:
            _shutdown(persistent)
            if restarted is not None:
                _shutdown(restarted)


if __name__ == '__main__':
    unittest.main()
