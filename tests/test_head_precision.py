import unittest
import sys
import types

import torch

# SelectionHead does not use madmom, but its shared modules file imports the
# optional package for LogSpectrogram.
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

from coda.models.heads import SelectionHead, SelectionHeadV2


class SelectionHeadPrecisionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 autocast")
    def test_fp16_film_convolution_overflow_is_prevented(self):
        head = SelectionHead(
            in_channels=128,
            roi_size=(4, 24),
            zdim=4,
            dropout=0.0,
            groupnorm=True,
        ).cuda().eval()
        with torch.no_grad():
            head.gamma.weight.zero_()
            head.gamma.bias.fill_(40.0)
            head.beta.weight.zero_()
            head.beta.bias.zero_()
            head.conv1.conv.weight.fill_(1.0)

        features = torch.full((1, 128, 8, 32), 200.0, device='cuda')
        rois = torch.tensor([[0.0, 0.0, 0.0, 31.0, 7.0]], device='cuda')
        z = torch.zeros(1, 4, device='cuda')

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            logits = head(features, rois, z, spatial_scale=1.0)

        self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 support is required",
    )
    def test_bf16_cross_attention_backward_runs_in_float32(self):
        head = SelectionHeadV2(
            in_channels=16,
            roi_size=(4, 8),
            zdim=8,
            dropout=0.0,
            groupnorm=True,
            audio_dim=8,
            num_heads=4,
            audio_window=6,
            attn_dropout=0.0,
        ).cuda().train()
        features = torch.randn(2, 16, 8, 16, device='cuda', requires_grad=True)
        rois = torch.tensor([
            [0.0, 0.0, 0.0, 7.0, 7.0],
            [1.0, 8.0, 0.0, 15.0, 7.0],
        ], device='cuda')
        z = torch.randn(2, 8, device='cuda', requires_grad=True)
        audio = torch.randn(2, 7, 8, device='cuda', requires_grad=True)
        lengths = torch.tensor([7, 4], device='cuda')

        seen_dtypes = []
        original_forward = head.cross_attn.forward

        def recording_forward(*args, **kwargs):
            query = kwargs['query'] if 'query' in kwargs else args[0]
            key = kwargs['key'] if 'key' in kwargs else args[1]
            seen_dtypes.append((query.dtype, key.dtype))
            return original_forward(*args, **kwargs)

        head.cross_attn.forward = recording_forward
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = head(
                features, rois, z, spatial_scale=1.0,
                audio_seq=audio, audio_lengths=lengths,
            )
            loss = logits.float().square().mean()
        loss.backward()

        self.assertEqual(seen_dtypes, [(torch.float32, torch.float32)])
        gradients = [
            parameter.grad for parameter in head.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(audio.grad).all())


if __name__ == '__main__':
    unittest.main()
