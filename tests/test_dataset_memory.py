import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from coda.dataset import _CompactSequenceBuilder
from coda.utils.data_utils import load_piece


def _sequence(frame, onset=False):
    return {
        'piece_id': 3,
        'is_onset': onset,
        'start_frame': 4,
        'frame': frame,
        'max_x_shift': (-12, 20),
        'max_y_shift': (-5, 7),
        'true_position': np.array([10, 20, 1, 2, 0], dtype=np.int32),
        'true_system': np.array([100, 50, 180, 40], dtype=np.float64),
        'true_bar': np.array([80, 50, 30, 40], dtype=np.float64),
        'height': 40.0,
        'scale_factor': 2.5,
        'gt_system_page_idx': 1,
        'gt_bar_in_system_idx': 2,
        'gt_valid': True,
        'prev_gt_system_page_idx': 0,
        'prev_gt_bar_page_idx': 1,
    }


class CompactSequenceStoreTest(unittest.TestCase):
    def test_rows_preserve_mapping_behavior_and_values(self):
        builder = _CompactSequenceBuilder()
        builder.add([_sequence(5), _sequence(6, onset=True)])
        store = builder.finish()

        self.assertEqual(len(store), 2)
        self.assertEqual(store[1]['frame'], 6)
        self.assertIs(store[1]['is_onset'], True)
        np.testing.assert_array_equal(
            store[0]['true_position'], np.array([10, 20, 1, 2, 0])
        )
        self.assertAlmostEqual(store[0].get('scale_factor'), 2.5)

        mutable_copy = dict(store[0])
        mutable_copy['start_frame'] = 99
        self.assertEqual(store[0]['start_frame'], 4)
        self.assertLess(store.nbytes, 512)

    def test_defaults_cover_optional_temporal_fields(self):
        sequence = _sequence(5)
        del sequence['prev_gt_system_page_idx']
        del sequence['prev_gt_bar_page_idx']
        builder = _CompactSequenceBuilder()
        builder.add([sequence])
        row = builder.finish()[0]
        self.assertEqual(row.get('prev_gt_system_page_idx'), -1)
        self.assertEqual(row.get('prev_gt_bar_page_idx'), -1)


class StreamedAudioSetupTest(unittest.TestCase):
    def test_load_piece_can_return_path_without_decoding_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            piece = 'example'
            np.savez(
                os.path.join(tmpdir, piece + '.npz'),
                sheets=np.full((1, 4, 6), 255, dtype=np.uint8),
                coords=np.array([{
                    'note_x': 1, 'onset': 0.0, 'system_idx': 0,
                    'page_nr': 0, 'bar_idx': 0,
                }], dtype=object),
                systems=np.array([{
                    'x': 3, 'y': 2, 'w': 4, 'h': 2, 'page_nr': 0,
                }], dtype=object),
                bars=np.array([{
                    'x': 3, 'y': 2, 'w': 2, 'h': 2, 'page_nr': 0,
                }], dtype=object),
                synthesized=np.array(False),
            )

            with mock.patch(
                'coda.utils.data_utils.load_wav',
                side_effect=AssertionError('full WAV decode must not occur'),
            ):
                result = load_piece(tmpdir, piece, load_audio=False)

            self.assertEqual(result[7], os.path.join(tmpdir, piece + '.wav'))


if __name__ == '__main__':
    unittest.main()
