import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from coda.augmentations.impulse_response import (
    ImpulseResponse,
    is_candidate_ir,
    load_signal,
)
from coda.utils.general import AverageMeter


class ImpulseResponseTests(unittest.TestCase):
    def _write_wav(self, path, seconds=0.1, sample_rate=22050):
        path.parent.mkdir(parents=True, exist_ok=True)
        n_samples = max(1, int(seconds * sample_rate))
        signal = np.zeros(n_samples, dtype=np.float32)
        signal[0] = 1.0
        sf.write(path, signal, sample_rate, subtype="FLOAT")

    def test_candidate_filter_rejects_non_ir_assets_and_long_audio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            genuine = root / "venue" / "mono" / "room_ir.wav"
            rendered = root / "venue" / "CONVOLUTION_EXAMPLES" / "music.wav"
            sweep = root / "venue" / "room_sweeps" / "capture.wav"
            long_audio = root / "venue" / "psalm.wav"
            self._write_wav(genuine)
            self._write_wav(rendered)
            self._write_wav(sweep)
            self._write_wav(long_audio, seconds=1.1)

            self.assertTrue(is_candidate_ir(genuine))
            self.assertFalse(is_candidate_ir(rendered))
            self.assertFalse(is_candidate_ir(sweep))
            self.assertFalse(is_candidate_ir(long_audio, max_duration=1.0))

    def test_loaded_ir_is_unit_l1_and_wet_audio_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "room_ir.wav"
            self._write_wav(path)
            _, ir, error = load_signal(path)
            self.assertIsNone(error)
            self.assertAlmostEqual(float(np.sum(np.abs(ir))), 1.0, places=6)

            augmentation = ImpulseResponse.__new__(ImpulseResponse)
            augmentation.ir_prob = 1.0
            augmentation.irs = [ir]
            dry = np.linspace(-0.25, 0.25, 4096, dtype=np.float32)
            wet = augmentation({"performance": dry.copy()})["performance"]

            self.assertEqual(wet.shape, dry.shape)
            self.assertEqual(wet.dtype, np.float32)
            self.assertTrue(np.isfinite(wet).all())
            self.assertLessEqual(
                float(np.max(np.abs(wet))),
                float(np.max(np.abs(dry))) + 1e-6,
            )


class AverageMeterTests(unittest.TestCase):
    def test_nonfinite_value_is_rejected(self):
        meter = AverageMeter()
        meter.update(1.0)
        with self.assertRaises(ValueError):
            meter.update(float("nan"))
        self.assertEqual(meter.avg, 1.0)


if __name__ == "__main__":
    unittest.main()
