"""
HiVa-DETR
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

D-FINE's hybrid encoder with HiVa's stride-2 value level: a channel projection per backbone
level, an intra-scale transformer on the listed levels, a CSP-ELAN feature pyramid across
levels, and one more top-down step to stride 2 whose output the decoder reads as values only.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...nn.blocks import ConvNormLayerFuse, RepNCSPELAN4, SCDown
from ...nn.position_encoding import build_2d_sincos_position_embedding
from ...nn.transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = ["HybridEncoder"]


@register()
class HybridEncoder(nn.Module):
    """
    Args:
        in_channels / feat_strides: the backbone levels, lowest stride first.
        hidden_dim: every level is projected to this width.
        use_encoder_idx / num_encoder_layers / nhead / dim_feedforward / dropout / enc_act /
            pe_temperature: the intra-scale transformer applied to the listed levels.
        expansion / depth_mult / act: width, depth and activation of the pyramid's fusion blocks.
        fine_in_channels / fine_dim: HiVa's value level P1, one stride finer than the pyramid,
            built from the backbone's stride-2 stem map (``HGNetv2(return_stem=True)``, the first
            of ``feats``, ``fine_in_channels`` wide) as one more top-down step of the pyramid at
            ``fine_dim`` channels: the stem map through a 1x1 (``fine_input_proj``), the finest
            pyramid level through a 1x1 and upsampled (``fine_lateral_conv``), the two
            concatenated and fused by a ``RepNCSPELAN4`` (``fine_fpn_block``). It stays out of the
            pyramid and the encoder's levels: the decoder reads it as values only.
            ``fine_in_channels`` 0: no value level.
        fine_expansion / fine_depth_mult: the width and depth factors of ``fine_fpn_block``.

    ``forward(feats)`` returns a dict with ``feats`` (the pyramid, one tensor per level) and, with
    a value level, ``fine``.
    """

    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        fine_in_channels=0,
        fine_dim=64,
        fine_expansion=0.5,
        fine_depth_mult=0.34,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.fine_in_channels = fine_in_channels

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hidden_dim)),
                        ]
                    )
                )
            )

        # intra-scale transformer, one per listed level
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=enc_act
        )
        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))]
        )

        fusion = dict(c3=hidden_dim * 2, c4=round(expansion * hidden_dim // 2), n=round(3 * depth_mult), act=act)
        # top-down: lateral 1x1 on the coarser level, upsample, fuse with the finer one
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.lateral_convs.append(ConvNormLayerFuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))
        # bottom-up: downsample the finer level, fuse with the coarser one
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2)))
            self.pan_blocks.append(RepNCSPELAN4(hidden_dim * 2, hidden_dim, **fusion))

        # the value level: one more top-down step at fine_dim channels
        if fine_in_channels > 0:
            self.fine_input_proj = ConvNormLayerFuse(fine_in_channels, fine_dim, 1, 1)
            self.fine_lateral_conv = ConvNormLayerFuse(hidden_dim, fine_dim, 1, 1)
            self.fine_fpn_block = RepNCSPELAN4(
                fine_dim * 2,
                fine_dim,
                c3=fine_dim * 2,
                c4=max(1, round(fine_expansion * fine_dim // 2)),
                n=max(1, round(3 * fine_depth_mult)),
                act=act,
            )

    @staticmethod
    def _upsample(x: torch.Tensor, size) -> torch.Tensor:
        """
        ``x`` upsampled bilinearly by 2 and cut to the finer level's ``size`` (a coarse map is the
        ceiling of the input over its stride, so twice it holds the finer map), so that every fine
        cell's centre is sampled where it lies in the coarse map.
        """
        h, w = size
        if 2 * x.shape[2] < h or 2 * x.shape[3] < w:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)[:, :, :h, :w]

    def forward(self, feats):
        stem = None
        if self.fine_in_channels > 0:
            stem, feats = feats[0], feats[1:]
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # intra-scale transformer
        for i, enc_ind in enumerate(self.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, HW, C]
            pos_embed = build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature)
            memory = self.encoder[i](src_flatten, pos_embed=pos_embed.to(src_flatten.device))
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # top-down: coarsest level first, each finer level fused with the upsampled result
        inner_outs = [proj_feats[-1]]
        for i, idx in enumerate(range(len(self.in_channels) - 1, 0, -1)):
            feat_high = self.lateral_convs[i](inner_outs[0])
            feat_low = proj_feats[idx - 1]
            inner_outs[0] = feat_high
            upsample_feat = self._upsample(feat_high, feat_low.shape[2:])
            inner_outs.insert(0, self.fpn_blocks[i](torch.concat([upsample_feat, feat_low], dim=1)))

        out = {}
        # the value level, one more top-down step from the finest pyramid level
        if stem is not None:
            semantics = self._upsample(self.fine_lateral_conv(inner_outs[0]), stem.shape[2:])
            out["fine"] = self.fine_fpn_block(torch.concat([semantics, self.fine_input_proj(stem)], dim=1))

        # bottom-up: finest level first, each coarser level fused with the downsampled result
        outs = [inner_outs[0]]
        for i in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[i](outs[-1])
            outs.append(self.pan_blocks[i](torch.concat([downsample_feat, inner_outs[i + 1]], dim=1)))

        out["feats"] = outs
        return out
