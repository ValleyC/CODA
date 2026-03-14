"""
Shared visual backbone with Feature Pyramid Network (FPN) and note regression head.

SharedBackbone produces multi-scale feature maps (P3 at stride 8, P4 at stride 16)
from score images conditioned on audio embeddings via FiLM layers.

NoteHead regresses note positions within bar regions using ROI-aligned features.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from torchvision.ops import roi_align

from coda.models.modules import Conv, Focus, FiLMConv, Bottleneck, Concat
from coda.utils.general import make_divisible


class SharedBackbone(nn.Module):
    """
    Shared visual backbone + FPN that produces P3 and P4 feature maps.

    Reuses parse_model() to build backbone+head layers
    (everything defined in YAML backbone + head sections, excluding Detect).

    Returns P3 (stride 8) and P4 (stride 16) feature maps.
    """

    def __init__(self, cfg):
        super().__init__()
        self.yaml = cfg
        self.zdim = cfg['encoder']['params']['zdim']

        # Build backbone + head layers (no Detect)
        self.model, self.save = self._build_layers(deepcopy(cfg))

        # Determine P3 and P4 indices by running a dummy forward pass
        # P4 = after first upsample+concat+bottleneck section (stride 16 features)
        # P3 = final output (stride 8 features, full FPN output)
        self._p3_idx = len(self.model) - 1  # Last layer = P3
        self._p4_idx = self._find_p4_idx()

    def _build_layers(self, cfg):
        """Build backbone + head layers from YAML config, excluding any Detect."""
        ch = [1]  # input channels (grayscale)
        anchors = cfg.get('anchors', [[1, 1]])
        nc = cfg.get('nc', 3)
        activation = eval(cfg.get('activation', 'nn.ELU'))
        groupnorm = cfg.get('groupnorm', False)
        zdim = cfg['encoder']['params']['zdim']
        audio_dim = cfg['encoder']['params'].get('hidden_size', 64)
        crossattn_cfg = cfg.get('crossattn', {})

        layers, save, c2 = [], [], ch[-1]
        all_layer_defs = cfg['backbone'] + cfg['head']

        for i, (f, m_str, args) in enumerate(all_layer_defs):
            m = eval(m_str) if isinstance(m_str, str) else m_str

            # Skip Detect layer
            if m is None or (isinstance(m_str, str) and 'Detect' in m_str):
                continue

            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass

            if m in [Conv, Focus, FiLMConv, Bottleneck]:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2, 8)
                args = [c1, c2, *args[1:]]
            elif m in [nn.BatchNorm2d, nn.GroupNorm]:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            else:
                c2 = ch[f]

            if m in [Conv, Focus, Bottleneck]:
                m_ = m(*args, groupnorm=groupnorm, activation=activation)
            elif m == FiLMConv:
                m_ = m(*args, zdim=zdim, groupnorm=groupnorm, activation=activation)
            else:
                m_ = m(*args)

            t = str(m)[8:-2].replace('__main__.', '')
            np_count = sum([x.numel() for x in m_.parameters()])
            m_.i, m_.f, m_.type, m_.np = i, f, t, np_count
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            ch.append(c2)

        model = nn.Sequential()
        for i, layer in enumerate(layers):
            model.add_module(f'{layer._get_name()}_{i}', layer)

        return model, sorted(save)

    def _find_p4_idx(self):
        """Find the P4 feature map index (after first upsample section).

        In the standard config, the head has:
        [FiLMConv, Upsample, Concat, Bottleneck, FiLMConv, Upsample, Concat, Bottleneck]
        P4 = output after the first Bottleneck (index 12 in backbone+head = 3 in head)
        """
        # Count backbone layers
        n_backbone = len(self.yaml['backbone'])
        # P4 is at backbone_len + 3 (FiLMConv + Upsample + Concat + Bottleneck)
        return n_backbone + 3

    def forward(self, x, z, audio_seq=None, audio_lengths=None):
        """
        Args:
            x: [B, 1, H, W] score image
            z: [B, zdim] audio conditioning
            audio_seq: [B, T, audio_dim] optional audio sequence for cross-attention
            audio_lengths: [B] optional sequence lengths

        Returns:
            p3: [B, C, H/8, W/8] stride-8 features
            p4: [B, C, H/16, W/16] stride-16 features
        """
        y = []
        p4 = None

        for idx, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if isinstance(m, FiLMConv):
                x = m(x, z)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)

            # Capture P4
            if idx == self._p4_idx:
                p4 = x

        p3 = x  # Last layer output
        return p3, p4


class NoteHead(nn.Module):
    """
    Note position regression head.

    Takes ROI-aligned features from bar regions and predicts (cx, cy)
    within the bar using sigmoid (guarantees output in [0, 1]).
    """

    def __init__(self, in_channels, roi_size=(6, 12), zdim=128,
                 groupnorm=True, activation=nn.ELU):
        super().__init__()
        self.roi_size = roi_size

        # FiLM conditioning
        self.gamma = nn.Linear(zdim, in_channels)
        self.beta = nn.Linear(zdim, in_channels)

        # Feature extraction
        mid_channels = in_channels // 2
        self.conv = Conv(in_channels, mid_channels, 3, groupnorm=groupnorm, activation=activation)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Position regression (sigmoid output -> [0, 1] within bar)
        self.pos_head = nn.Linear(mid_channels, 2)

        # Confidence head
        self.conf_head = nn.Linear(mid_channels, 1)

    def forward(self, p3, bar_rois_page, z, spatial_scale=1.0/8):
        """
        Args:
            p3: [B, C, H/8, W/8] P3 feature map
            bar_rois_page: [N, 5] ROI boxes in page pixel coords [batch_idx, x1, y1, x2, y2]
            z: [B, zdim] audio conditioning
            spatial_scale: Scale factor from pixel to feature space (1/stride)

        Returns:
            positions: [N, 2] (cx, cy) in [0, 1] within bar (sigmoid output)
            confidences: [N, 1] confidence in [0, 1]
        """
        if bar_rois_page.shape[0] == 0:
            return (torch.zeros(0, 2, device=p3.device),
                    torch.zeros(0, 1, device=p3.device))

        # ROI align
        roi_features = roi_align(p3, bar_rois_page, output_size=self.roi_size,
                                  spatial_scale=spatial_scale)  # [N, C, H_roi, W_roi]

        # FiLM conditioning (expand z per ROI based on batch index)
        batch_indices = bar_rois_page[:, 0].long()
        z_per_roi = z[batch_indices]  # [N, zdim]
        gamma = self.gamma(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z_per_roi).unsqueeze(-1).unsqueeze(-1)
        roi_features = gamma * roi_features + beta

        # Feature extraction
        x = self.conv(roi_features)  # [N, mid_C, H_roi, W_roi]
        x = self.pool(x).flatten(1)  # [N, mid_C]

        # Position: sigmoid guarantees [0, 1] -- note is within bar by construction
        positions = torch.sigmoid(self.pos_head(x))  # [N, 2]

        # Confidence
        confidences = torch.sigmoid(self.conf_head(x))  # [N, 1]

        return positions, confidences
