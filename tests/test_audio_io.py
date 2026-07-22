import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from coda.utils.general import load_wav_segment


class AudioSegmentTests(unittest.TestCase):
    def test_direct_segment_read_matches_full_msmd_style_wav(self):
        sample_rate = 22050
        rng = np.random.default_rng(7)
        original = rng.uniform(-0.8, 0.8, sample_rate * 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "piece.wav")
            sf.write(path, original, sample_rate, subtype="PCM_16")
            full, _ = sf.read(path, dtype="float32", always_2d=False)
            segment = load_wav_segment(path, 1234, 45678, sample_rate)

        np.testing.assert_array_equal(segment, full[1234:45678])

    def test_segment_bounds_are_clamped(self):
        sample_rate = 22050
        original = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "piece.wav")
            sf.write(path, original, sample_rate, subtype="PCM_16")
            segment = load_wav_segment(path, -100, 100, sample_rate)
        self.assertEqual(segment.shape, (100,))


if __name__ == "__main__":
    unittest.main()
