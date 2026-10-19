"""
HGNetv2 (PP-HGNetV2), the backbone of D-FINE and HiVa-DETR.

reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...core import register

__all__ = ["HGNetv2"]


class StageConfig(NamedTuple):
    """One HGNetv2 stage, as listed in ``HGNetv2.arch_configs``."""

    in_channels: int
    mid_channels: int
    out_channels: int
    num_blocks: int
    downsample: bool
    light_block: bool
    kernel_size: int
    layer_num: int


class LearnableAffineBlock(nn.Module):
    """``scale * x + bias`` with a learnable scalar each; follows every activation when ``use_lab`` is on."""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, x):
        return torch.addcmul(self.bias.to(x.dtype), x, self.scale.to(x.dtype))


class ConvBNAct(nn.Module):
    """Conv -> BN -> ReLU (optional) -> LAB (optional), with 'same'-style padding for odd kernels."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1, use_act=True, use_lab=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.ReLU() if use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if use_act and use_lab else nn.Identity()

    def forward(self, x):
        return self.lab(self.act(self.bn(self.conv(x))))


class LightConvBNAct(nn.Module):
    """A 1x1 pointwise conv (no activation) followed by a depthwise kxk conv."""

    def __init__(self, in_chs, out_chs, kernel_size, use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs, use_act=True, use_lab=use_lab)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    """The stride-4 stem: a strided 3x3, a two-branch 2x2 / max-pool split, a strided 3x3 and a 1x1."""

    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def half(self, x):
        """The stride-2 map, ``2 * mid_chs`` channels: the two branches concatenated, before the second strided conv."""
        x = self.stem1(x)
        # the 2x2 convs and the pool are padded by one on the bottom/right so they keep the size
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        x1 = self.pool(x)
        return torch.cat([x1, x2], dim=1)

    def finish(self, half):
        """The stride-4 output from the stride-2 map."""
        return self.stem4(self.stem3(half))


class HGBlock(nn.Module):
    """
    The HGNet block: ``layer_num`` convs applied in sequence, their outputs concatenated with
    the input and aggregated back to ``out_chs`` (a squeeze / excite pair of 1x1 convs), with a
    residual connection when ``residual`` is set.
    """

    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            chs_in = in_chs if i == 0 else mid_chs
            if light_block:
                self.layers.append(LightConvBNAct(chs_in, mid_chs, kernel_size=kernel_size, use_lab=use_lab))
            else:
                self.layers.append(ConvBNAct(chs_in, mid_chs, kernel_size=kernel_size, stride=1, use_lab=use_lab))

        total_chs = in_chs + layer_num * mid_chs
        self.aggregation = nn.Sequential(
            ConvBNAct(total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab),
            ConvBNAct(out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab),
        )

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = self.aggregation(torch.cat(output, dim=1))
        if self.residual:
            x = x + identity
        return x


class HGStage(nn.Module):
    """An optional depthwise stride-2 downsample followed by ``block_num`` HG blocks (residual after the first)."""

    def __init__(
        self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False
    ):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2, groups=in_chs, use_act=False, use_lab=use_lab
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                HGBlock(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=i > 0,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                )
                for i in range(block_num)
            ]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))


@register()
class HGNetv2(nn.Module):
    """
    HGNetv2 at strides 4, 8, 16 and 32, returning the stages listed in ``return_idx``.

    Args:
        name: the architecture, ``B0`` .. ``B6``.
        use_lab: add a LearnableAffineBlock after every activation.
        return_idx: which stages (0-based) to return.
        return_stem: also return the stem's stride-2 map (twice the stem's middle width) ahead of
            the stages, from which the encoder builds HiVa's stride-2 value level.
    """

    # stem: [in, mid, out] channels; stages: see StageConfig
    arch_configs = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": StageConfig(16, 16, 64, 1, False, False, 3, 3),
                "stage2": StageConfig(64, 32, 256, 1, True, False, 3, 3),
                "stage3": StageConfig(256, 64, 512, 2, True, True, 5, 3),
                "stage4": StageConfig(512, 128, 1024, 1, True, True, 5, 3),
            },
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": StageConfig(32, 32, 96, 1, False, False, 3, 4),
                "stage2": StageConfig(96, 64, 384, 1, True, False, 3, 4),
                "stage3": StageConfig(384, 128, 768, 3, True, True, 5, 4),
                "stage4": StageConfig(768, 256, 1536, 1, True, True, 5, 4),
            },
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                "stage1": StageConfig(48, 48, 128, 1, False, False, 3, 6),
                "stage2": StageConfig(128, 96, 512, 1, True, False, 3, 6),
                "stage3": StageConfig(512, 192, 1024, 3, True, True, 5, 6),
                "stage4": StageConfig(1024, 384, 2048, 1, True, True, 5, 6),
            },
        },
    }

    def __init__(self, name, use_lab=False, return_idx=(1, 2, 3), return_stem=False):
        super().__init__()
        self.return_idx = return_idx
        self.return_stem = return_stem

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]

        self.stem = StemBlock(
            in_chs=stem_channels[0], mid_chs=stem_channels[1], out_chs=stem_channels[2], use_lab=use_lab
        )

        self.stages = nn.ModuleList()
        for i, cfg in enumerate(stage_config.values()):
            if i > max(self.return_idx):
                break
            self.stages.append(
                HGStage(
                    cfg.in_channels,
                    cfg.mid_channels,
                    cfg.out_channels,
                    cfg.num_blocks,
                    cfg.layer_num,
                    downsample=cfg.downsample,
                    light_block=cfg.light_block,
                    kernel_size=cfg.kernel_size,
                    use_lab=use_lab,
                )
            )

    def forward(self, x):
        half = self.stem.half(x)
        x = self.stem.finish(half)
        outs = [half] if self.return_stem else []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
