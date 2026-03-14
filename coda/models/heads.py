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

        # ROI align at known locations
        roi_features = roi_align(p3, rois, output_size=self.roi_size,
                                  spatial_scale=spatial_scale)  # [N, C, H_roi, W_roi]

        # FiLM conditioning per ROI
        batch_indices = rois[:, 0].long()
        z_per_roi = z[batch_indices]
        gamma = self.gamma(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        roi_features = gamma * roi_features + beta

        # Conv + pool -> logit
        x = self.conv1(roi_features)
        x = self.conv2(x)
        x = self.pool(x).flatten(1)  # [N, C]
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)  # [N]

        return logits


class SelectionHeadV2(nn.Module):
    """
    Enhanced selection head with cross-attention and optional candidate context.

    Pipeline: ROI Align -> FiLM(z) -> Conv -> Conv -> CrossAttn(roi, audio_seq) -> Pool -> FC
    Falls back to FiLM-only when audio_seq is None (streaming warmup / backward compat).
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
            audio_seq: [B, T, audio_dim] per-frame Mamba hidden states (None = FiLM fallback)
            audio_lengths: [B] actual lengths for masking

        Returns:
            logits: [N] scalar logit per candidate
        """
        if rois.shape[0] == 0:
            return torch.zeros(0, device=features.device)

        N = rois.shape[0]
        C = features.shape[1]
        batch_indices = rois[:, 0].long()

        # Step 1: ROI Align
        roi_features = roi_align(features, rois, output_size=self.roi_size,
                                 spatial_scale=spatial_scale)

        # Step 2: FiLM conditioning
        z_per_roi = z[batch_indices]
        gamma = self.gamma(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        roi_features = gamma * roi_features + beta

        # Step 3: Conv layers
        x = self.conv1(roi_features)
        x = self.conv2(x)

        # Step 4: Cross-attention (skip if audio_seq not available)
        if audio_seq is not None:
            H_roi, W_roi = x.shape[2], x.shape[3]

            # Window audio to last audio_window frames
            audio = audio_seq
            a_lengths = audio_lengths
            if self.audio_window is not None and audio.shape[1] > self.audio_window:
                audio = audio[:, -self.audio_window:, :]
                if a_lengths is not None:
                    a_lengths = torch.clamp(a_lengths, max=self.audio_window)
            T = audio.shape[1]

            # Project audio and index per-ROI
            audio_kv = self.audio_proj(audio)           # [B, T, C]
            audio_kv_per_roi = audio_kv[batch_indices]  # [N, T, C]

            # Flatten spatial dims as queries
            x_flat = x.permute(0, 2, 3, 1).reshape(N, H_roi * W_roi, C)
            x_query = self.attn_ln(x_flat)

            # Key padding mask for variable-length audio
            key_padding_mask = None
            if a_lengths is not None:
                a_lengths_per_roi = a_lengths[batch_indices]
                key_padding_mask = (
                    torch.arange(T, device=audio.device).unsqueeze(0)
                    >= a_lengths_per_roi.unsqueeze(1)
                )

            # Cross-attention
            x_attn, _ = self.cross_attn(
                query=x_query,
                key=audio_kv_per_roi,
                value=audio_kv_per_roi,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            # Output projection + layer scale + residual
            x_attn = self.attn_out_proj(x_attn)
            x_attn = x_attn * self.layer_scale
            x_attn = x_attn.reshape(N, H_roi, W_roi, C).permute(0, 3, 1, 2)
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
