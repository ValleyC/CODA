"""
Selection heads for candidate scoring in the CODA cascade.

SelectionHead: FiLM-only selection head (ROI-align, FiLM conditioning, conv, pool, FC).
SelectionHeadV2: Enhanced head with FiLM + cross-attention over audio sequence.

These are used by both the system selection and bar selection stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from coda.models.modules import Conv
from coda.utils.sequence import take_last_valid_window


def _fp16_autocast_active(tensor):
    """Return whether CUDA autocast is currently using narrow-range FP16."""
    autocast_dtype = (
        torch.get_autocast_dtype('cuda')
        if hasattr(torch, 'get_autocast_dtype')
        else torch.get_autocast_gpu_dtype()
    )
    return (
        tensor.is_cuda
        and torch.is_autocast_enabled()
        and autocast_dtype == torch.float16
    )


def _cuda_autocast_active(tensor):
    """Return whether a CUDA tensor is executing inside an AMP region."""
    return tensor.is_cuda and torch.is_autocast_enabled()


def _roi_film_first_conv(features, rois, z, roi_size, spatial_scale,
                          gamma_layer, beta_layer, conv_layer):
    """Apply the numerically sensitive ROI-FiLM-first-conv prefix.

    Learned FiLM gains can make otherwise finite ROI features large enough
    that the 3x3 convolution's FP16 accumulation overflows before GroupNorm
    can rescale it. BF16 has sufficient exponent range; for explicit FP16 AMP,
    keep only this prefix in FP32 and leave the rest of the head AMP eligible.
    """
    batch_indices = rois[:, 0].long()
    if _fp16_autocast_active(features):
        with torch.autocast(device_type='cuda', enabled=False):
            roi_features = roi_align(
                features.float(), rois.float(), output_size=roi_size,
                spatial_scale=spatial_scale,
            )
            z_per_roi = z[batch_indices].float()
            gamma = gamma_layer(z_per_roi).unsqueeze(-1).unsqueeze(-1)
            beta = beta_layer(z_per_roi).unsqueeze(-1).unsqueeze(-1)
            return conv_layer(gamma * roi_features + beta), batch_indices

    roi_features = roi_align(
        features, rois, output_size=roi_size, spatial_scale=spatial_scale
    )
    z_per_roi = z[batch_indices]
    gamma = gamma_layer(z_per_roi).unsqueeze(-1).unsqueeze(-1)
    beta = beta_layer(z_per_roi).unsqueeze(-1).unsqueeze(-1)
    return conv_layer(gamma * roi_features + beta), batch_indices


class SelectionHead(nn.Module):
    """
    Classifies which candidate ROI is active, conditioned on audio.

    ROI-align -> FiLM -> 2x Conv -> AdaptiveAvgPool -> Linear -> scalar logit per candidate.
    log_softmax over candidates gives log p(candidate_i).
    """

    def __init__(self, in_channels, roi_size, zdim=128, dropout=0.1,
                 groupnorm=True, activation=nn.ELU):
        super().__init__()
        self.roi_size = tuple(roi_size)

        # FiLM conditioning
        self.gamma = nn.Linear(zdim, in_channels)
        self.beta = nn.Linear(zdim, in_channels)

        # Feature extraction
        self.conv1 = Conv(in_channels, in_channels, 3, groupnorm=groupnorm, activation=activation)
        self.conv2 = Conv(in_channels, in_channels, 3, groupnorm=groupnorm, activation=activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, p3, rois, z, spatial_scale=1.0/8):
        """
        Args:
            p3: [B, C, H/8, W/8] feature map
            rois: [N, 5] ROI boxes [batch_idx, x1, y1, x2, y2] in pixel space
            z: [B, zdim] audio conditioning

        Returns:
            logits: [N] scalar logit per candidate
        """
        if rois.shape[0] == 0:
            return torch.zeros(0, device=p3.device)

        # ROI alignment, FiLM, and the first convolution form one numerical
        # boundary; the helper protects its FP16 accumulation when necessary.
        x, _ = _roi_film_first_conv(
            p3, rois, z, self.roi_size, spatial_scale,
            self.gamma, self.beta, self.conv1,
        )

        # Remaining conv + pool -> logit
        x = self.conv2(x)
        x = self.pool(x).flatten(1)  # [N, C]
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)  # [N]

        return logits


class SelectionHeadV2(nn.Module):
    """
    Enhanced selection head with cross-attention and optional candidate context.

    Pipeline: ROI Align -> FiLM(z) -> Conv -> Conv -> CrossAttn(roi, audio_seq) -> Pool -> FC.
    Before an audio history is available, the head uses its FiLM-conditioned
    visual representation directly.
    """

    def __init__(self, in_channels, roi_size, zdim=128, dropout=0.1,
                 groupnorm=True, activation=nn.ELU,
                 # Cross-attention params
                 audio_dim=64, num_heads=4, layer_scale_init=0.1,
                 audio_window=64, attn_dropout=0.1,
                 # Candidate context params
                 use_candidate_context=False):
        super().__init__()
        self.roi_size = tuple(roi_size)
        self.audio_window = audio_window
        self.use_candidate_context = use_candidate_context
        C = in_channels

        # FiLM conditioning (always present)
        self.gamma = nn.Linear(zdim, C)
        self.beta = nn.Linear(zdim, C)

        # Feature extraction
        self.conv1 = Conv(C, C, 3, groupnorm=groupnorm, activation=activation)
        self.conv2 = Conv(C, C, 3, groupnorm=groupnorm, activation=activation)

        # Cross-attention: ROI spatial features (queries) attend to audio (keys/values)
        self.audio_proj = nn.Linear(audio_dim, C)
        self.attn_ln = nn.LayerNorm(C)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=C, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.layer_scale = nn.Parameter(torch.ones(C) * layer_scale_init)
        self.attn_out_proj = nn.Linear(C, C)

        # Pool + scoring
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        fc_in = 2 * C if use_candidate_context else C
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, features, rois, z, spatial_scale=1.0/8,
                audio_seq=None, audio_lengths=None):
        """
        Args:
            features: [B, C, H/s, W/s] feature map
            rois: [N, 5] ROI boxes [batch_idx, x1, y1, x2, y2]
            z: [B, zdim] pooled audio conditioning
            spatial_scale: 1/stride
            audio_seq: [B, T, audio_dim] per-frame Mamba hidden states, or None
                before the streaming history is initialized
            audio_lengths: [B] actual lengths for masking

        Returns:
            logits: [N] scalar logit per candidate
        """
        if rois.shape[0] == 0:
            return torch.zeros(0, device=features.device)

        N = rois.shape[0]
        C = features.shape[1]
        # Steps 1-3: protect the potentially overflowing FiLM/conv prefix for
        # FP16 while retaining ordinary AMP execution for BF16 and FP32.
        x, batch_indices = _roi_film_first_conv(
            features, rois, z, self.roi_size, spatial_scale,
            self.gamma, self.beta, self.conv1,
        )
        x = self.conv2(x)

        # Step 4: Cross-attention once streaming audio history is available.
        if audio_seq is not None:
            H_roi, W_roi = x.shape[2], x.shape[3]

            # Window relative to each sample's valid length. A batch-wide
            # right slice would select padding/post-padding states for shorter
            # sequences in a variable-length batch.
            audio, a_lengths = take_last_valid_window(
                audio_seq, audio_lengths, self.audio_window
            )
            T = audio.shape[1]

            # Key padding mask for variable-length audio
            key_padding_mask = None
            if a_lengths is not None:
                a_lengths_per_roi = a_lengths[batch_indices]
                key_padding_mask = (
                    torch.arange(T, device=audio.device).unsqueeze(0)
                    >= a_lengths_per_roi.unsqueeze(1)
                )

            # Attention uses FP32 under AMP; the bounded temporal window keeps
            # this block small while preserving stable Q/K/V gradients. Cast
            # the residual back so the backbone and scorer remain AMP eligible.
            if _cuda_autocast_active(features):
                residual_dtype = x.dtype
                with torch.autocast(device_type='cuda', enabled=False):
                    audio_kv = self.audio_proj(audio.float())
                    audio_kv_per_roi = audio_kv[batch_indices]
                    x_flat = x.float().permute(0, 2, 3, 1).reshape(
                        N, H_roi * W_roi, C
                    )
                    x_query = self.attn_ln(x_flat)
                    # Use the deterministic math SDPA kernel for this bounded
                    # attention block.
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=True,
                        enable_mem_efficient=False,
                    ):
                        x_attn, _ = self.cross_attn(
                            query=x_query,
                            key=audio_kv_per_roi,
                            value=audio_kv_per_roi,
                            key_padding_mask=key_padding_mask,
                            need_weights=False,
                        )
                    x_attn = self.attn_out_proj(x_attn)
                    x_attn = x_attn * self.layer_scale
                    x_attn = x_attn.reshape(
                        N, H_roi, W_roi, C
                    ).permute(0, 3, 1, 2)
                    x = (x.float() + x_attn).to(residual_dtype)
            else:
                audio_kv = self.audio_proj(audio)
                audio_kv_per_roi = audio_kv[batch_indices]
                x_flat = x.permute(0, 2, 3, 1).reshape(
                    N, H_roi * W_roi, C
                )
                x_query = self.attn_ln(x_flat)
                x_attn, _ = self.cross_attn(
                    query=x_query,
                    key=audio_kv_per_roi,
                    value=audio_kv_per_roi,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                x_attn = self.attn_out_proj(x_attn)
                x_attn = x_attn * self.layer_scale
                x_attn = x_attn.reshape(
                    N, H_roi, W_roi, C
                ).permute(0, 3, 1, 2)
                x = x + x_attn

        # Step 5: Pool
        x = self.pool(x).flatten(1)
        x = self.dropout(x)

        # Step 6: Candidate context (mean of all candidates per batch item)
        if self.use_candidate_context:
            B = z.shape[0]
            context = torch.zeros(B, C, device=x.device)
            counts = torch.zeros(B, 1, device=x.device)
            context.scatter_add_(0, batch_indices.unsqueeze(1).expand_as(x), x)
            counts.scatter_add_(0, batch_indices.unsqueeze(1),
                                torch.ones(N, 1, device=x.device))
            counts = counts.clamp(min=1)
            context = context / counts
            context_per_roi = context[batch_indices]
            x = torch.cat([x, context_per_roi], dim=1)

        # Step 7: Score
        logits = self.fc(x).squeeze(-1)
        return logits
