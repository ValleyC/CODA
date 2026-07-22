import unittest
import sys
import types

import torch

# The local unit-test environment intentionally omits madmom; NoteHead does
# not use it, but coda.models.modules imports these symbols for LogSpectrogram.
try:
    import madmom  # noqa: F401
except ImportError:
    madmom_module = types.ModuleType('madmom')
    audio_module = types.ModuleType('madmom.audio')
    stft_module = types.ModuleType('madmom.audio.stft')
    spectrogram_module = types.ModuleType('madmom.audio.spectrogram')
    stft_module.fft_frequencies = lambda *args, **kwargs: None
    spectrogram_module.LogarithmicFilterbank = object
    sys.modules.update({
        'madmom': madmom_module,
        'madmom.audio': audio_module,
        'madmom.audio.stft': stft_module,
        'madmom.audio.spectrogram': spectrogram_module,
    })

from coda.models.backbone import NoteHead


class NoteHeadTests(unittest.TestCase):
    def test_strict_state_loading_rejects_unknown_parameters(self):
        head = NoteHead(in_channels=8, roi_size=(2, 2), zdim=4,
                        groupnorm=False)
        state = dict(head.state_dict())
        state['unexpected.weight'] = torch.randn(1, 4)

        with self.assertRaises(RuntimeError):
            head.load_state_dict(state, strict=True)

    def test_empty_rois_return_positions_only(self):
        head = NoteHead(in_channels=8, roi_size=(2, 2), zdim=4,
                        groupnorm=False)
        positions = head(
            torch.zeros(1, 8, 4, 4),
            torch.zeros(0, 5),
            torch.zeros(1, 4),
        )

        self.assertIsInstance(positions, torch.Tensor)
        self.assertEqual(tuple(positions.shape), (0, 2))


if __name__ == '__main__':
    unittest.main()
