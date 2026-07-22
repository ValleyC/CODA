import random
import unittest

import numpy as np
import torch

from coda.utils.general import (
    isolated_python_numpy_seed,
    isolated_python_numpy_torch_seed,
    stable_named_seed,
)


class DeterministicSeedTests(unittest.TestCase):
    def test_named_seed_is_stable_and_piece_specific(self):
        first = stable_named_seed(42, "piece_a")
        self.assertEqual(first, stable_named_seed(42, "piece_a"))
        self.assertNotEqual(first, stable_named_seed(42, "piece_b"))
        self.assertNotEqual(first, stable_named_seed(43, "piece_a"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2 ** 32)

    def test_isolated_seed_repeats_locally_and_restores_callers(self):
        random.seed(1234)
        np.random.seed(5678)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        with isolated_python_numpy_seed(99):
            first = (random.random(), np.random.random())
        with isolated_python_numpy_seed(99):
            second = (random.random(), np.random.random())
        self.assertEqual(first, second)

        actual_after = (random.random(), np.random.random())
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        expected_after = (random.random(), np.random.random())
        self.assertEqual(actual_after, expected_after)

    def test_all_cpu_rngs_are_isolated_and_repeatable(self):
        random.seed(5)
        np.random.seed(6)
        torch.manual_seed(7)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        def draw():
            with isolated_python_numpy_torch_seed(123):
                return random.random(), np.random.random(), torch.rand(3)

        first = draw()
        second = draw()
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[1].tolist(), numpy_state[1].tolist())
        torch.testing.assert_close(torch.get_rng_state(), torch_state)


if __name__ == "__main__":
    unittest.main()
