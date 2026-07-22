"""
Mamba-based audio encoder for streaming score following.

Processes spectrogram frames through a lightweight per-frame encoder
followed by stacked Mamba SSM layers. Supports both parallel training
(full sequence forward) and O(1) streaming inference (step-by-step).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from coda.models.modules import Flatten, TemporalBatchNorm

# Try to import Mamba
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class MambaConditioning(nn.Module):
    """
    Native Mamba conditioning with per-frame processing.

    Key features:
    - No 40-frame chunking - processes each frame directly
    - Lightweight per-frame encoder
    - Training: forward() on full sequence (parallel)
    - Inference: step() per frame (O(1) streaming)
    """

    def __init__(self, zdim=128, n_mamba_layers=2, activation=nn.ELU,
                 freq_dim=78, hidden_size=64, groupnorm=False,
                 d_state=16, d_conv=4, expand=2, dropout=0.1,
                 encoder_type='linear', normalize_input=False,
                 use_fast_path=True):
        super(MambaConditioning, self).__init__()

        if not HAS_MAMBA:
            raise ImportError("MambaConditioning requires mamba-ssm package")

        self.zdim = zdim
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim
        self.n_mamba_layers = n_mamba_layers
        self.normalize_input = normalize_input
        self.use_fast_path = use_fast_path

        if isinstance(activation, str):
            activation = eval(activation)

        # Input normalization
        # Use LayerNorm instead of TemporalBatchNorm to avoid:
        # 1) Running stats mismatch between training and inference
        # 2) Noisy stats from per-sequence normalization (batch=1)
        if normalize_input:
            self.input_norm = nn.LayerNorm(freq_dim)
        else:
            self.input_norm = None

        # Per-frame encoder (lightweight)
        if encoder_type == 'linear':
            self.frame_encoder = nn.Sequential(
                nn.Linear(freq_dim, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if groupnorm else nn.BatchNorm1d(hidden_size * 2),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size) if groupnorm else nn.BatchNorm1d(hidden_size),
                activation(),
            )
        elif encoder_type == 'conv1d':
            self.frame_encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.GroupNorm(1, 32) if groupnorm else nn.BatchNorm1d(32),
                activation(),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(1, 64) if groupnorm else nn.BatchNorm1d(64),
                activation(),
                nn.Conv1d(64, hidden_size, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(1, hidden_size) if groupnorm else nn.BatchNorm1d(hidden_size),
                activation(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.encoder_type = encoder_type

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                # Retain Mamba's parallel fused implementation for training;
                # the recurrence itself runs in FP32 below, while the model's
                # conditioning-boundary hook protects it from an exploding
                # downstream Jacobian.
                use_fast_path=use_fast_path,
            )
            for _ in range(n_mamba_layers)
        ])
        self.mamba_dropout = nn.Dropout(dropout)
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_mamba_layers)])

        # Z encoder
        self.z_enc = nn.Sequential(
            nn.Linear(hidden_size, zdim),
            nn.LayerNorm(zdim) if groupnorm else nn.BatchNorm1d(zdim),
            activation()
        )

        # Inference state
        self._mamba_conv_states = None
        self._mamba_ssm_states = None
        self._mamba_cached_output = None  # For cross-attention buffer

    def _encode_frames(self, x):
        """Encode spectrogram frames. x: [B, T, freq_dim] -> [B, T, hidden_size]"""
        B, T, F = x.shape
        if self.encoder_type == 'linear':
            x_flat = x.reshape(B * T, F)
            encoded = self.frame_encoder(x_flat)
            return encoded.reshape(B, T, -1)
        else:
            x_flat = x.reshape(B * T, 1, F)
            encoded = self.frame_encoder(x_flat)
            return encoded.reshape(B, T, -1)

    def encode_sequence(self, specs, hidden=None):
        """Training: encode full sequences with parallel Mamba forward()."""
        lengths = [s.shape[0] for s in specs]
        # Apply input normalization BEFORE padding
        # LayerNorm normalizes per-frame, no batch statistics to contaminate
        if self.input_norm is not None:
            specs = [self.input_norm(s) for s in specs]

        # Pad after normalization. ``pad_sequence`` performs the copies in one
        # vectorized operation instead of launching one slice assignment per
        # sample, while preserving autograd into the normalized inputs.
        padded = pad_sequence(specs, batch_first=True)

        # Encode frames. The dense frame projection remains AMP eligible; it is
        # a conventional feed-forward block and accounts for most of the audio
        # encoder's arithmetic on long sequences.
        encoded = self._encode_frames(padded)

        # Selective scans are recurrent numerical accumulations. With long
        # (up to two-minute) training windows, running them in FP16 can produce
        # a sample-local NaN even though both the waveform and spectrogram are
        # finite. Keep the state-space recurrence and final conditioning
        # projection in FP32 while allowing the surrounding visual network and
        # frame projection to benefit from AMP.
        with torch.autocast(device_type=encoded.device.type, enabled=False):
            mamba_out = encoded.float()
            for mamba_layer, norm in zip(self.mamba_layers, self.mamba_norms):
                residual = mamba_out
                mamba_out = mamba_layer(mamba_out)
                mamba_out = self.mamba_dropout(mamba_out)
                mamba_out = norm(mamba_out + residual)

            # Gather every sample's last valid state in one indexed operation.
            lengths_tensor = torch.as_tensor(lengths, device=mamba_out.device)
            rows = torch.arange(len(lengths), device=mamba_out.device)
            z = self.z_enc(mamba_out[rows, lengths_tensor - 1])

        return z, mamba_out, lengths_tensor

    def _init_mamba_states(self, batch_size, device, dtype):
        """Initialize Mamba states for step mode."""
        self._mamba_conv_states = []
        self._mamba_ssm_states = []
        for mamba_layer in self.mamba_layers:
            conv_state = torch.zeros(batch_size, mamba_layer.d_inner, mamba_layer.d_conv, device=device, dtype=dtype)
            ssm_state = torch.zeros(batch_size, mamba_layer.d_inner, mamba_layer.d_state, device=device, dtype=dtype)
            self._mamba_conv_states.append(conv_state)
            self._mamba_ssm_states.append(ssm_state)

    def get_conditioning(self, spec_frame, hidden=None, update_every_frame=True):
        """Streaming inference - updates EVERY frame with O(1) Mamba step()."""
        if spec_frame.dim() == 1:
            spec_frame = spec_frame.unsqueeze(0)

        # Reset on hidden=None (page change)
        if hidden is None:
            self._mamba_conv_states = None
            self._mamba_ssm_states = None

        # Initialize states
        if self._mamba_conv_states is None:
            self._init_mamba_states(1, spec_frame.device, spec_frame.dtype)

        # Apply input normalization if enabled
        # LayerNorm works directly on [1, freq_dim]
        if self.input_norm is not None:
            spec_frame = self.input_norm(spec_frame)

        # Encode single frame
        encoded = self._encode_frames(spec_frame.unsqueeze(1))

        # Mamba step (O(1) per frame)
        mamba_out = encoded
        for i, (mamba_layer, norm) in enumerate(zip(self.mamba_layers, self.mamba_norms)):
            residual = mamba_out
            mamba_out, self._mamba_conv_states[i], self._mamba_ssm_states[i] = \
                mamba_layer.step(mamba_out, self._mamba_conv_states[i], self._mamba_ssm_states[i])
            mamba_out = self.mamba_dropout(mamba_out)
            mamba_out = norm(mamba_out + residual)

        # Cache Mamba output for cross-attention buffer
        self._mamba_cached_output = mamba_out.squeeze(1).detach()

        z = self.z_enc(mamba_out.squeeze(1))
        return z, True

    def get_cached_output(self):
        """Return cached Mamba output for cross-attention buffer."""
        return self._mamba_cached_output

    def reset_inference_state(self):
        """Reset inference state for new piece."""
        self._mamba_conv_states = None
        self._mamba_ssm_states = None
        self._mamba_cached_output = None
