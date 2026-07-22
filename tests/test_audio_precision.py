import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


# audio_encoder imports two helper classes that are irrelevant to the Mamba
# precision contract, while their source module imports the optional madmom
# package. Define only those unused names so this focused test remains runnable
# in the lightweight CPU test environment.
_module_stub = types.ModuleType('coda.models.modules')
_module_stub.Flatten = nn.Flatten
_module_stub.TemporalBatchNorm = nn.BatchNorm1d
with patch.dict(sys.modules, {'coda.models.modules': _module_stub}):
    audio_encoder = importlib.import_module('coda.models.audio_encoder')


class _RecordingMamba(nn.Module):
    seen_dtypes = []
    seen_fast_paths = []

    def __init__(self, d_model, d_state, d_conv, expand, use_fast_path=True):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.seen_fast_paths.append(use_fast_path)

    def forward(self, value):
        self.seen_dtypes.append(value.dtype)
        return value


class TestAudioPrecisionBoundary(unittest.TestCase):
    def test_mamba_recurrence_promotes_amp_projection_to_float32(self):
        _RecordingMamba.seen_dtypes.clear()
        _RecordingMamba.seen_fast_paths.clear()
        with patch.object(audio_encoder, 'HAS_MAMBA', True), \
                patch.object(audio_encoder, 'Mamba', _RecordingMamba, create=True):
            encoder = audio_encoder.MambaConditioning(
                zdim=8,
                n_mamba_layers=2,
                freq_dim=4,
                hidden_size=4,
                groupnorm=True,
                dropout=0.0,
            )

        # Model an AMP-eligible frame projection without requiring a CUDA
        # device in the unit test. encode_sequence must promote this tensor at
        # the selective-scan boundary.
        encoder._encode_frames = lambda padded: torch.ones(
            padded.shape[0], padded.shape[1], 4, dtype=torch.float16
        )
        specs = [torch.ones(3, 4), torch.ones(2, 4)]

        z, audio_sequence, lengths = encoder.encode_sequence(specs)

        self.assertEqual(_RecordingMamba.seen_dtypes, [torch.float32, torch.float32])
        self.assertEqual(_RecordingMamba.seen_fast_paths, [True, True])
        self.assertEqual(audio_sequence.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float32)
        self.assertEqual(lengths.tolist(), [3, 2])


if __name__ == '__main__':
    unittest.main()
