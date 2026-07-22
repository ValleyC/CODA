import unittest

import torch

from coda.utils.sequence import take_last_valid_window


class LastValidWindowTests(unittest.TestCase):
    def test_window_is_relative_to_each_samples_valid_length(self):
        sequence = torch.arange(2 * 10, dtype=torch.float32).reshape(2, 10, 1)
        lengths = torch.tensor([10, 4])

        windowed, window_lengths = take_last_valid_window(sequence, lengths, 3)

        torch.testing.assert_close(windowed[0, :, 0], torch.tensor([7.0, 8.0, 9.0]))
        torch.testing.assert_close(windowed[1, :, 0], torch.tensor([11.0, 12.0, 13.0]))
        torch.testing.assert_close(window_lengths, torch.tensor([3, 3]))

    def test_short_valid_sequence_is_left_aligned_and_maskable(self):
        sequence = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

        windowed, window_lengths = take_last_valid_window(
            sequence, torch.tensor([2]), 4
        )

        torch.testing.assert_close(windowed[0, :, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(window_lengths, torch.tensor([2]))

    def test_unpadded_batch_uses_fast_right_slice(self):
        sequence = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        windowed, lengths = take_last_valid_window(sequence, None, 2)
        torch.testing.assert_close(windowed[0, :, 0], torch.tensor([4.0, 5.0]))
        self.assertIsNone(lengths)


if __name__ == "__main__":
    unittest.main()
