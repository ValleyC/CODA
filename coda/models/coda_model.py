"""
CODA: Selection-Based Hierarchical Score Following.

Architecture:
    Shared Backbone + FPN -> P3 (stride 8)
    Stage 1: SelectionHead on ROI-aligned P3 at ALL known system boxes -> log p(system_i)
    Stage 2: SelectionHead on ROI-aligned P3 at bars in selected system -> log p(bar_j | system_i)
    Stage 3: NoteHead on ROI-aligned P3 at selected bar -> sigmoid (cx, cy)

No anchors, no NMS, no objectness -- pure classification over known candidates.
"""

import json
import math
import os
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coda.models.backbone import SharedBackbone, NoteHead
from coda.models.modules import LogSpectrogram
from coda.models.audio_encoder import MambaConditioning
from coda.models.heads import SelectionHead, SelectionHeadV2
from coda.models.builder import initialize_weights
from coda.utils.data_utils import FPS, SAMPLE_RATE, FRAME_SIZE


class SelectionCascadeModel(nn.Module):
    """
    Hierarchical score following via selection among known candidates.

    Stage 1: Classify which system on the page is active
    Stage 2: Classify which bar within the selected system is active
    Stage 3: Regress note position within the selected bar
    """

    def __init__(self, cfg):
        super().__init__()
        self.yaml = cfg

        cascade_cfg = cfg['cascade_config']
        zdim = cfg['encoder']['params']['zdim']
        groupnorm = cfg.get('groupnorm', True)
        activation = eval(cfg.get('activation', 'nn.ELU'))
        dropout = cascade_cfg.get('selection_dropout', 0.1)

        # --- Shared components ---
        self.spec_module = LogSpectrogram(sr=SAMPLE_RATE, fps=FPS, frame_size=FRAME_SIZE)
        self.conditioning_network = eval(cfg['encoder']['type'])(
            groupnorm=groupnorm, **cfg['encoder']['params']
        )
        self.backbone = SharedBackbone(cfg)

        # --- Stage 1: System selection ---
        system_roi_size = tuple(cascade_cfg['system_roi_size'])
        head_v2_cfg = cascade_cfg.get('head_v2', {})
        use_cross_attn = head_v2_cfg.get('use_cross_attn', False)
        audio_dim = cfg['encoder']['params'].get('hidden_size', 64)

        if use_cross_attn:
            self.system_head = SelectionHeadV2(
                in_channels=128, roi_size=system_roi_size, zdim=zdim,
                dropout=dropout, groupnorm=groupnorm, activation=activation,
                audio_dim=audio_dim,
                num_heads=head_v2_cfg.get('num_heads', 4),
                layer_scale_init=head_v2_cfg.get('layer_scale_init', 0.1),
                audio_window=head_v2_cfg.get('audio_window', 64),
                attn_dropout=head_v2_cfg.get('attn_dropout', 0.1),
                use_candidate_context=head_v2_cfg.get('use_candidate_context', False),
            )
        else:
            self.system_head = SelectionHead(
                in_channels=128, roi_size=system_roi_size, zdim=zdim,
                dropout=dropout, groupnorm=groupnorm, activation=activation,
            )

        # --- Stage 2: Bar selection ---
        bar_roi_size = tuple(cascade_cfg['bar_roi_size'])
        bar_v2_cfg = cascade_cfg.get('bar_head_v2', {})
        bar_use_cross_attn = bar_v2_cfg.get('use_cross_attn', False)

        if bar_use_cross_attn:
            self.bar_head = SelectionHeadV2(
                in_channels=128, roi_size=bar_roi_size, zdim=zdim,
                dropout=dropout, groupnorm=groupnorm, activation=activation,
                audio_dim=audio_dim,
                num_heads=bar_v2_cfg.get('num_heads', 4),
                layer_scale_init=bar_v2_cfg.get('layer_scale_init', 0.1),
                audio_window=bar_v2_cfg.get('audio_window', 64),
                attn_dropout=bar_v2_cfg.get('attn_dropout', 0.1),
                use_candidate_context=bar_v2_cfg.get('use_candidate_context', False),
            )
        else:
            self.bar_head = SelectionHead(
                in_channels=128, roi_size=bar_roi_size, zdim=zdim,
                dropout=dropout, groupnorm=groupnorm, activation=activation,
            )

        # --- Stage 3: Note regression ---
        note_roi_size = tuple(cascade_cfg['note_roi_size'])
        self.note_head = NoteHead(
            in_channels=128, roi_size=note_roi_size, zdim=zdim,
            groupnorm=groupnorm, activation=activation,
        )

        # --- Beam search config ---
        self.top_k_systems = cascade_cfg.get('top_k_systems', 3)
        self.top_m_bars = cascade_cfg.get('top_m_bars', 3)

        # --- Break mode config (inference only) ---
        break_cfg = cascade_cfg.get('break_mode', {})
        self.break_mode_enabled = break_cfg.get('enabled', False)
        self.break_onset_threshold = break_cfg.get('onset_threshold', 0.1)
        self.break_release_threshold = break_cfg.get('release_threshold', 0.25)
        self.break_energy_window = break_cfg.get('energy_window', 100)
        self.break_min_history = break_cfg.get('min_history', 10)
        self.break_silence_onset = break_cfg.get('silence_onset_frames', 3)
        self.break_grace_frames = break_cfg.get('grace_frames', 8)
        self.break_prior_scale = break_cfg.get('prior_scale', 0.0)
        self.break_beam_k = break_cfg.get('beam_k_systems', -1)
        self.break_beam_m = break_cfg.get('beam_m_bars', 3)

        # Learnable temporal priors (bounded nn.Parameters)
        # "same"/"stay" is always 0.0 (anchored); others are <= 0 penalties.
        # Raw params are unconstrained; clamped to [-8.0, 0.0] at access time.
        sys_init = cascade_cfg.get('system_transition', {})
        self._sys_tp_raw = nn.Parameter(torch.tensor([
            sys_init.get('adjacent', -1.0),
            sys_init.get('far', -3.0),
        ]))

        bar_init = cascade_cfg.get('bar_transition', {})
        self._bar_tp_raw = nn.Parameter(torch.tensor([
            bar_init.get('forward_1', -0.3),
            bar_init.get('forward_2', -1.5),
            bar_init.get('backward_1', -2.0),
            bar_init.get('far', -3.0),
        ]))

        # Keep dict form for backward compat (used by test scripts for overrides)
        self.system_transition = {
            'same': 0.0,
            'adjacent': sys_init.get('adjacent', -1.0),
            'far': sys_init.get('far', -3.0),
        }
        self.bar_transition = {
            'stay': 0.0,
            'forward_1': bar_init.get('forward_1', -0.3),
            'forward_2': bar_init.get('forward_2', -1.5),
            'backward_1': bar_init.get('backward_1', -2.0),
            'far': bar_init.get('far', -3.0),
        }

        # --- Tracking state ---
        self.prev_system_idx = None
        self.prev_bar_idx = None

        # --- Break mode state (inference only) ---
        self._break_mode_active = False
        self._in_silence = False
        self._silent_frame_count = 0
        self._grace_frames_remaining = 0
        self._energy_history = []

        # --- Initialize weights ---
        self.apply(initialize_weights)

        # --- Compute backbone strides ---
        self._init_strides()

    def _init_strides(self):
        dummy_input = torch.zeros(1, 1, 128, 128)
        dummy_z = torch.zeros(1, self.yaml['encoder']['params']['zdim'])
        with torch.no_grad():
            p3, p4 = self.backbone(dummy_input, dummy_z)
        self.p3_stride = 128 // p3.shape[-1]

    def _call_head(self, head, features, rois, z, spatial_scale,
                   audio_seq=None, audio_lengths=None):
        """Dispatch to SelectionHead or SelectionHeadV2 with correct args."""
        if isinstance(head, SelectionHeadV2):
            return head(features, rois, z, spatial_scale,
                        audio_seq=audio_seq, audio_lengths=audio_lengths)
        else:
            return head(features, rois, z, spatial_scale)

    def compute_spec(self, x, tempo_aug=False):
        return self.spec_module(x, tempo_aug)

    def encode_sequence(self, perf, hidden=None):
        return self.conditioning_network.encode_sequence(perf, hidden)

    def reset_tracking_state(self):
        self.prev_system_idx = None
        self.prev_bar_idx = None
        self.reset_break_mode()

    def reset_break_mode(self):
        self._break_mode_active = False
        self._in_silence = False
        self._silent_frame_count = 0
        self._grace_frames_remaining = 0
        self._energy_history.clear()

    @property
    def is_break_mode(self):
        return self._break_mode_active

    def update_break_mode(self, sig_excerpt):
        """Update break mode using waveform RMS energy with hysteresis.

        Args:
            sig_excerpt: raw waveform numpy array (the audio chunk for this frame)
        Returns:
            dict: break mode diagnostics {is_break_mode, in_silence,
                  grace_frames_remaining, norm_energy}
        """
        diag = {'is_break_mode': False, 'in_silence': False,
                'grace_frames_remaining': 0, 'norm_energy': 0.0}
        if not self.break_mode_enabled:
            return diag

        # RMS energy from waveform (not spectrogram)
        raw_energy = float(np.sqrt(np.mean(sig_excerpt ** 2)) + 1e-10)

        # Freeze energy history while in silence -- prevents self-normalization
        # where sustained silence shifts the baseline and triggers false release.
        if not self._in_silence:
            self._energy_history.append(raw_energy)
            if len(self._energy_history) > self.break_energy_window:
                self._energy_history.pop(0)

        # Disable detection until enough history -- no fallback to raw
        if len(self._energy_history) < self.break_min_history:
            diag['norm_energy'] = -1.0  # sentinel: not ready
            return diag

        # Normalize against pre-silence baseline: z-score -> sigmoid -> [0, 1]
        mu = sum(self._energy_history) / len(self._energy_history)
        var = sum((x - mu) ** 2 for x in self._energy_history) / len(self._energy_history)
        sigma = math.sqrt(var) + 1e-8
        z = (raw_energy - mu) / sigma
        norm_energy = 1.0 / (1.0 + math.exp(-z))

        # Hysteresis state machine
        if self._in_silence:
            if norm_energy > self.break_release_threshold:
                # Release event: exit silence, set grace counter
                # +1 so countdown effectively starts NEXT frame
                self._in_silence = False
                self._silent_frame_count = 0
                if self._break_mode_active:
                    self._grace_frames_remaining = self.break_grace_frames + 1
        else:
            if norm_energy < self.break_onset_threshold:
                self._silent_frame_count += 1
                if self._silent_frame_count >= self.break_silence_onset:
                    self._in_silence = True
                    self._break_mode_active = True
            else:
                self._silent_frame_count = 0

        # Grace countdown (after audio resumes)
        if self._break_mode_active and not self._in_silence:
            self._grace_frames_remaining -= 1
            if self._grace_frames_remaining <= 0:
                self._break_mode_active = False

        diag.update({
            'is_break_mode': self._break_mode_active,
            'in_silence': self._in_silence,
            'grace_frames_remaining': self._grace_frames_remaining,
            'norm_energy': norm_energy,
        })
        return diag

    def _xywh_to_rois(self, boxes_xywh, batch_idx, device):
        """Convert [N, 4] xywh boxes to [N, 5] ROI format [batch_idx, x1, y1, x2, y2]."""
        if boxes_xywh.shape[0] == 0:
            return torch.zeros(0, 5, device=device)
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        batch_col = torch.full((boxes_xywh.shape[0], 1), batch_idx, device=device, dtype=boxes_xywh.dtype)
        return torch.cat([batch_col, x1.unsqueeze(1), y1.unsqueeze(1),
                          x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)

    def forward(self, score, perf, system_boxes, bar_boxes, bars_per_system,
                gt_system_idx=None, gt_bar_in_sys=None,
                prev_gt_system_idx=None, prev_gt_bar_page_idx=None,
                tempo_aug=False, p_pred=0.0):
        """
        Forward pass with known candidate boxes.

        During training (gt_system_idx/gt_bar_in_sys provided):
          - System logits are computed over ALL systems (always correct for CE loss)
          - Bar/note stages route through GT system by default
          - With p_pred > 0 (scheduled sampling), some samples route through
            predicted system. Bar targets are remapped; if GT bar is not in
            predicted system's candidates, bar/note loss is masked.

        During inference (gt indices not provided):
          - Use argmax for system/bar selection

        Args:
            score: [B, 1, H, W] score image
            perf: list of audio signals
            system_boxes: list of [N_sys_i, 4] xywh tensors per batch item
            bar_boxes: list of [N_bar_i, 4] xywh tensors per batch item
            bars_per_system: list of list-of-lists per batch item
            gt_system_idx: [B] LongTensor -- GT system index (for training)
            gt_bar_in_sys: [B] LongTensor -- GT bar-in-system index (for training)
            tempo_aug: audio tempo augmentation
            p_pred: probability of using predicted routing (scheduled sampling)

        Returns:
            dict with system_logits, bar_logits, note_positions, etc.
            When p_pred > 0, also includes:
              'bar_target_remapped': [B] remapped bar targets for loss
              'ss_bar_valid': [B] bool mask (False = mask bar/note loss)
        """
        B, _, H, W = score.shape
        device = score.device
        assert (gt_system_idx is None) == (gt_bar_in_sys is None), \
            "gt_system_idx and gt_bar_in_sys must both be provided or both be None"
        training = gt_system_idx is not None

        # Audio encoding
        perf = self.compute_spec(perf, tempo_aug)
        z, audio_seq, audio_lengths = self.encode_sequence(perf)

        # Shared backbone -> P3 features
        p3, p4 = self.backbone(score, z, audio_seq, audio_lengths)
        spatial_scale = 1.0 / self.p3_stride

        # --- Stage 1: System selection ---
        # Score ALL systems on each page (no routing needed)
        all_sys_rois = []
        sys_counts = []
        for b in range(B):
            boxes_b = system_boxes[b].to(device)
            rois_b = self._xywh_to_rois(boxes_b, b, device)
            all_sys_rois.append(rois_b)
            sys_counts.append(boxes_b.shape[0])

        all_sys_rois = torch.cat(all_sys_rois, dim=0) if all_sys_rois else torch.zeros(0, 5, device=device)
        sys_logits = self._call_head(self.system_head, p3, all_sys_rois, z, spatial_scale,
                                     audio_seq=audio_seq, audio_lengths=audio_lengths)

        # Add temporal priors to system logits during training (Phase 2a)
        if prev_gt_system_idx is not None:
            sys_logits = self._add_system_temporal_priors_train(
                sys_logits, sys_counts, prev_gt_system_idx, B)

        sys_log_probs = self._grouped_log_softmax(sys_logits, sys_counts)

        # --- Stage 2: Bar selection ---
        # Route through GT system, predicted system (scheduled sampling), or
        # argmax (inference).
        all_bar_rois = []
        bar_counts = []
        # Track actual system used and remapped bar targets for scheduled sampling
        actual_sys_idx = []  # which system was used per batch item
        bar_target_remapped = torch.zeros(B, dtype=torch.long, device=device) if training else None
        ss_bar_valid = torch.ones(B, dtype=torch.bool, device=device) if training else None

        offset = 0
        for b in range(B):
            n_sys = sys_counts[b]
            bps = bars_per_system[b]

            if training:
                # Scheduled sampling: use predicted system with probability p_pred
                use_pred = p_pred > 0 and _random.random() < p_pred
                if use_pred:
                    sys_lp_b = sys_log_probs[offset:offset + n_sys]
                    sel_sys = sys_lp_b.argmax().item() if n_sys > 0 else 0
                else:
                    sel_sys = gt_system_idx[b].item()

                actual_sys_idx.append(sel_sys)

                # Remap bar target when routing through non-GT system
                gt_sys = gt_system_idx[b].item()
                gt_bar_local = gt_bar_in_sys[b].item()

                if sel_sys == gt_sys:
                    # Same system -- bar target unchanged
                    bar_target_remapped[b] = gt_bar_local
                else:
                    # Different system -- find GT bar in predicted system's candidates
                    gt_bars_list = bps[gt_sys] if gt_sys < len(bps) else []
                    gt_bar_page = gt_bars_list[gt_bar_local] if gt_bar_local < len(gt_bars_list) else -1
                    pred_bars_list = bps[sel_sys] if sel_sys < len(bps) else []

                    if gt_bar_page >= 0 and gt_bar_page in pred_bars_list:
                        bar_target_remapped[b] = pred_bars_list.index(gt_bar_page)
                    else:
                        # GT bar not in predicted system -- mask bar/note loss
                        bar_target_remapped[b] = 0  # dummy, will be masked
                        ss_bar_valid[b] = False
            else:
                sys_lp_b = sys_log_probs[offset:offset + n_sys]
                sel_sys = sys_lp_b.argmax().item() if n_sys > 0 else 0
                actual_sys_idx.append(sel_sys)

            bar_indices = bps[sel_sys] if sel_sys < len(bps) else []
            all_bar_boxes_b = bar_boxes[b].to(device)

            if len(bar_indices) > 0:
                sel_bar_boxes = all_bar_boxes_b[bar_indices]
            else:
                sel_bar_boxes = all_bar_boxes_b

            rois_b = self._xywh_to_rois(sel_bar_boxes, b, device)
            all_bar_rois.append(rois_b)
            bar_counts.append(sel_bar_boxes.shape[0])

            offset += n_sys

        all_bar_rois = torch.cat(all_bar_rois, dim=0) if all_bar_rois else torch.zeros(0, 5, device=device)
        bar_logits = self._call_head(self.bar_head, p3, all_bar_rois, z, spatial_scale,
                                     audio_seq=audio_seq, audio_lengths=audio_lengths)

        # Add temporal priors to bar logits during training (Phase 2a)
        # Use actual_sys_idx (not gt_system_idx) so priors match the routed system
        if prev_gt_bar_page_idx is not None and training:
            actual_sys_tensor = torch.tensor(actual_sys_idx, dtype=torch.long, device=device)
            bar_logits = self._add_bar_temporal_priors_train(
                bar_logits, bar_counts, prev_gt_bar_page_idx,
                bars_per_system, actual_sys_tensor, B)

        bar_log_probs = self._grouped_log_softmax(bar_logits, bar_counts)

        # --- Stage 3: Note regression ---
        # Route through GT/remapped bar (training) or predicted bar (inference)
        note_rois = []
        offset = 0
        for b in range(B):
            n_bars = bar_counts[b]

            if training:
                sel_bar = bar_target_remapped[b].item()
                # Bounds check -- fallback to bar 0 if out of range
                if n_bars > 0 and sel_bar >= n_bars:
                    sel_bar = 0
                elif n_bars == 0:
                    sel_bar = 0
            else:
                bar_lp_b = bar_log_probs[offset:offset + n_bars]
                sel_bar = bar_lp_b.argmax().item() if n_bars > 0 else 0

            if n_bars > 0:
                note_rois.append(all_bar_rois[offset + sel_bar].unsqueeze(0))
            else:
                dummy_roi = torch.tensor([[b, 0, 0, W, H]], device=device, dtype=torch.float32)
                note_rois.append(dummy_roi)

            offset += n_bars

        note_rois = torch.cat(note_rois, dim=0)
        note_positions, note_confidences = self.note_head(
            p3, note_rois, z, spatial_scale=spatial_scale
        )

        result = {
            'sys_logits': sys_logits,
            'sys_log_probs': sys_log_probs,
            'sys_counts': sys_counts,
            'bar_logits': bar_logits,
            'bar_log_probs': bar_log_probs,
            'bar_counts': bar_counts,
            'note_positions': note_positions,
            'note_confidences': note_confidences,
            'note_rois': note_rois,
            'z': z,
        }
        # Scheduled sampling outputs
        if bar_target_remapped is not None:
            result['bar_target_remapped'] = bar_target_remapped
            result['ss_bar_valid'] = ss_bar_valid
        return result

    def _add_system_temporal_priors_train(self, sys_logits, sys_counts, prev_sys, B):
        """Add learnable temporal prior biases to system logits (training only).

        Builds a prior correction tensor and adds it to raw logits, so the CE
        loss trains the model to produce logits that work with temporal priors.
        Gradients flow back to the learnable prior parameters.
        """
        priors = self.sys_temporal_priors  # [adjacent, far], differentiable
        tp = torch.zeros_like(sys_logits)
        offset = 0
        for b in range(B):
            n = sys_counts[b]
            prev = prev_sys[b].item()
            if prev >= 0:  # -1 = no previous (first frame / page change)
                for i in range(n):
                    dist = abs(i - prev)
                    if dist == 1:
                        tp[offset + i] = priors[0]   # adjacent
                    elif dist > 1:
                        tp[offset + i] = priors[1]   # far
            offset += n
        return sys_logits + tp

    def _add_bar_temporal_priors_train(self, bar_logits, bar_counts,
                                       prev_bar_page, bars_per_system,
                                       gt_system_idx, B):
        """Add learnable temporal prior biases to bar logits (training only).

        Uses page-local bar indices to compute transition distances, matching
        the inference-time temporal prior computation.
        """
        priors = self.bar_temporal_priors  # [fwd1, fwd2, bwd1, far], differentiable
        tp = torch.zeros_like(bar_logits)
        offset = 0
        for b in range(B):
            n = bar_counts[b]
            prev = prev_bar_page[b].item()
            if prev >= 0 and n > 0:
                sel_sys = gt_system_idx[b].item()
                bps = bars_per_system[b]
                bar_indices = bps[sel_sys] if sel_sys < len(bps) else []
                for i in range(min(n, len(bar_indices))):
                    diff = bar_indices[i] - prev
                    if diff == 1:
                        tp[offset + i] = priors[0]   # forward_1
                    elif diff == 2:
                        tp[offset + i] = priors[1]   # forward_2
                    elif diff == -1:
                        tp[offset + i] = priors[2]   # backward_1
                    elif diff != 0:
                        tp[offset + i] = priors[3]   # far
            offset += n
        return bar_logits + tp

    def _grouped_log_softmax(self, logits, counts):
        """Apply log_softmax within each group defined by counts."""
        if logits.shape[0] == 0:
            return logits

        result = torch.zeros_like(logits)
        offset = 0
        for count in counts:
            if count > 0:
                result[offset:offset + count] = F.log_softmax(logits[offset:offset + count], dim=0)
            offset += count
        return result

    # --- Inference methods ---

    def inference_forward(self, p3, z, system_boxes_xywh, bar_boxes_xywh, bars_per_system,
                          audio_seq=None, audio_lengths=None):
        """
        Single-frame inference (no batching, no teacher forcing).

        Args:
            p3: [1, C, H/8, W/8] features
            z: [1, zdim] conditioning
            system_boxes_xywh: [N_sys, 4] all system boxes on current page
            bar_boxes_xywh: [N_bar, 4] all bar boxes on current page
            bars_per_system: list of lists
            audio_seq: [1, T, audio_dim] buffered audio hidden states (None = FiLM-only)
            audio_lengths: [1] actual buffer length

        Returns:
            dict with predictions and beam search results
        """
        device = p3.device
        spatial_scale = 1.0 / self.p3_stride

        # Stage 1: Score all systems
        sys_rois = self._xywh_to_rois(system_boxes_xywh, 0, device)
        sys_logits = self._call_head(self.system_head, p3, sys_rois, z, spatial_scale,
                                     audio_seq=audio_seq, audio_lengths=audio_lengths)
        sys_log_probs = F.log_softmax(sys_logits, dim=0)

        # Top-k systems for beam search (widen during break mode)
        if self._break_mode_active:
            bk = self.break_beam_k
            k = sys_log_probs.shape[0] if bk < 0 else min(bk, sys_log_probs.shape[0])
        else:
            k = min(self.top_k_systems, sys_log_probs.shape[0])
        top_sys_lp, top_sys_idx = torch.topk(sys_log_probs, k)

        # Stage 2: For each top system, score its bars
        paths = []
        for si in range(k):
            sys_i = top_sys_idx[si].item()
            sys_lp = top_sys_lp[si].item()
            sys_tp = self._system_temporal_prior(sys_i)
            if self._break_mode_active:
                sys_tp *= self.break_prior_scale

            bar_indices = bars_per_system[sys_i] if sys_i < len(bars_per_system) else []
            if not bar_indices:
                continue

            bar_boxes_in_sys = bar_boxes_xywh[bar_indices]
            bar_rois = self._xywh_to_rois(bar_boxes_in_sys, 0, device)
            bar_logits = self._call_head(self.bar_head, p3, bar_rois, z, spatial_scale,
                                         audio_seq=audio_seq, audio_lengths=audio_lengths)
            bar_lp = F.log_softmax(bar_logits, dim=0)

            if self._break_mode_active:
                m = min(self.break_beam_m, bar_lp.shape[0])
            else:
                m = min(self.top_m_bars, bar_lp.shape[0])
            top_bar_lp, top_bar_local_idx = torch.topk(bar_lp, m)

            for bi in range(m):
                bar_local = top_bar_local_idx[bi].item()
                bar_page_idx = bar_indices[bar_local]
                bar_score = top_bar_lp[bi].item()
                bar_tp = self._bar_temporal_prior(bar_page_idx)
                if self._break_mode_active:
                    bar_tp *= self.break_prior_scale

                # Note regression for this bar
                bar_roi = bar_rois[bar_local].unsqueeze(0)
                note_pos, _ = self.note_head(p3, bar_roi, z, spatial_scale=spatial_scale)
                note_cx, note_cy = note_pos[0].tolist()

                # Note confidence is NOT trained, so exclude from scoring
                joint_score = sys_lp + sys_tp + bar_score + bar_tp

                # Map note to page coords
                bar_box = bar_boxes_xywh[bar_page_idx]
                bar_x1 = (bar_box[0] - bar_box[2] / 2).item()
                bar_y1 = (bar_box[1] - bar_box[3] / 2).item()
                bar_w = bar_box[2].item()
                bar_h = bar_box[3].item()
                page_x = bar_x1 + note_cx * bar_w
                page_y = bar_y1 + note_cy * bar_h

                paths.append({
                    'system_idx': sys_i,
                    'bar_page_idx': bar_page_idx,
                    'bar_local_idx': bar_local,
                    'note_cx': note_cx,
                    'note_cy': note_cy,
                    'page_x': page_x,
                    'page_y': page_y,
                    'score': joint_score,
                    'sys_lp': sys_lp,
                    'bar_lp': bar_score,
                })

        if not paths:
            return None

        best = max(paths, key=lambda p: p['score'])

        # Freeze state during silence -- predictions are noisy without audio evidence.
        # During grace window (audio resumed, break still active): commit normally.
        if not self._in_silence:
            self.prev_system_idx = best['system_idx']
            self.prev_bar_idx = best['bar_page_idx']

        return {
            'best_path': best,
            'all_paths': paths,
            'sys_log_probs': sys_log_probs.cpu().numpy(),
            'note_page_x': best['page_x'],
            'note_page_y': best['page_y'],
        }

    @property
    def sys_temporal_priors(self):
        """Bounded system temporal priors: [adjacent, far] in [-8, 0]."""
        return self._sys_tp_raw.clamp(min=-8.0, max=0.0)

    @property
    def bar_temporal_priors(self):
        """Bounded bar temporal priors: [fwd1, fwd2, bwd1, far] in [-8, 0]."""
        return self._bar_tp_raw.clamp(min=-8.0, max=0.0)

    def _system_temporal_prior(self, sys_idx):
        if self.prev_system_idx is None:
            return 0.0
        dist = abs(sys_idx - self.prev_system_idx)
        if dist == 0:
            return 0.0  # "same" always anchored at 0
        elif dist == 1:
            return self.sys_temporal_priors[0].item()  # adjacent
        else:
            return self.sys_temporal_priors[1].item()  # far

    def _bar_temporal_prior(self, bar_idx):
        if self.prev_bar_idx is None:
            return 0.0
        diff = bar_idx - self.prev_bar_idx
        if diff == 0:
            return 0.0  # "stay" always anchored at 0
        elif diff == 1:
            return self.bar_temporal_priors[0].item()  # forward_1
        elif diff == 2:
            return self.bar_temporal_priors[1].item()  # forward_2
        elif diff == -1:
            return self.bar_temporal_priors[2].item()  # backward_1
        else:
            return self.bar_temporal_priors[3].item()  # far


def build_model(config, loss_calibration='static', label_smoothing=0.0):
    """Build a SelectionCascadeModel and its loss criterion from config."""
    from coda.utils.loss import selection_loss, UncertaintyWeightedLoss
    model = SelectionCascadeModel(config)
    if loss_calibration == 'uncertainty':
        criterion = UncertaintyWeightedLoss(
            label_smoothing=label_smoothing,
        )
    else:
        criterion = selection_loss
    return model, criterion


def load_model(param_path):
    """Load a pretrained SelectionCascadeModel from a checkpoint directory."""
    param_dir = os.path.dirname(param_path)
    with open(os.path.join(param_dir, 'net_config.json'), 'r') as f:
        config = json.load(f)

    model, criterion = build_model(config)
    try:
        model.load_state_dict(torch.load(param_path, map_location='cpu'))
    except:
        model = nn.parallel.DataParallel(model)
        model.load_state_dict(torch.load(param_path, map_location='cpu'))
        model = model.module

    return model, criterion
