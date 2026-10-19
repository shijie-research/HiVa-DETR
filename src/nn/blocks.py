"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The conv blocks of the hybrid encoder's feature pyramid: the conv-BN layers and the CSP / GELAN
fusion blocks.
"""

import torch
import torch.nn as nn

from .backbone.common import ConvNormLayer, get_activation

__all__ = ["CSPLayer", "ConvNormLayerFuse", "RepNCSPELAN4", "SCDown", "VGGBlock"]


class ConvNormLayerFuse(nn.Module):
    """Conv -> BN -> act."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    """Spatial-channel decoupled downsampling: a 1x1 pointwise conv then a strided depthwise conv."""

    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayerFuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayerFuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """RepVGG block: parallel 3x3 and 1x1 conv-BN branches."""

    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class CSPLayer(nn.Module):
    """Cross-stage partial layer: a stack of bottlenecks on one 1x1 branch, summed with a plain 1x1 branch."""

    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayerFuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayerFuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[VGGBlock(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayerFuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    """The GELAN fusion block (from YOLOv9): split, two chained CSP stages, concat everything, 1x1 out."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayerFuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act),
            ConvNormLayerFuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act),
            ConvNormLayerFuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayerFuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
