"""
HiVa-DETR
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

Multi-scale deformable attention (Deformable DETR), in the pure-PyTorch form D-FINE uses, with
HiVa's high-resolution value level: every query samples ``num_points`` locations per head on
every level around its reference box and sums the sampled values with learned weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.nn.init as init

__all__ = ["MSDeformableAttention", "ms_deformable_attention_core"]


def ms_deformable_attention_core(value, value_spatial_shapes, sampling_locations, attention_weights, num_points_list):
    """
    Args:
        value: one ``[bs, n_head, c, h*w]`` tensor per level.
        value_spatial_shapes: the ``(h, w)`` of each level.
        sampling_locations: ``[bs, query_length, n_head, sum(num_points), 2]`` in [0, 1].
        attention_weights: ``[bs, query_length, n_head, sum(num_points)]``.
        num_points_list: points sampled per level.

    Returns:
        ``[bs, query_length, n_head * c]``
    """
    bs, n_head, c, _ = value[0].shape
    _, len_q, _, _, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, len_q, sum(num_points_list))
    attn_weights_list = attn_weights.split(num_points_list, dim=-1)

    output = None
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_value_l = F.grid_sample(
            value_l, sampling_locations_list[level], mode="bilinear", padding_mode="zeros", align_corners=False
        )
        weighted = (sampling_value_l * attn_weights_list[level]).sum(-1)  # [bs * n_head, c, len_q]
        output = weighted if output is None else output + weighted

    return output.reshape(bs, n_head * c, len_q).permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    """
    Multi-scale deformable attention. ``num_points`` is one int for every level or a list with
    one entry per level; ``offset_scale`` scales the predicted offsets relative to the reference
    box size.

    ``fine_dim`` (0: off) makes the first level HiVa's high-resolution value level: a
    ``fine_dim``-channel map (not ``embed_dim``, not split across heads) that every head samples
    whole at its own points. Every sample goes through a per-head two-layer MLP (``fine_dim`` to
    ``fine_dim``, SiLU, ``fine_dim`` to ``head_dim``, no biases) before the weighted sum, which
    is added to what the head read from the other levels. Sampling before lifting keeps the dense
    map at ``fine_dim`` channels and runs the MLP on the samples only.
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, offset_scale=0.5, fine_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, "num_points needs one entry per level"
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]
        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.fine_dim = fine_dim
        if fine_dim > 0:
            # the per-head MLP lift of the fine samples, without biases so that a sample outside
            # the map (zero) contributes nothing (SiLU(0) = 0)
            self.fine_lift = nn.Parameter(torch.empty(num_heads, fine_dim, fine_dim))
            self.fine_lift2 = nn.Parameter(torch.empty(num_heads, fine_dim, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        if self.fine_dim > 0:
            for head in range(self.num_heads):
                init.xavier_uniform_(self.fine_lift.data[head])
                init.xavier_uniform_(self.fine_lift2.data[head])

    def _read_fine(self, value, hw, sampling_locations, attention_weights):
        """
        The fine level ``value`` ``[bs, c, h * w]`` sampled at ``sampling_locations``
        ``[bs, len_q, heads, P, 2]`` (in [0, 1]), every sample lifted by its head's MLP, and the
        lifted samples summed with ``attention_weights`` ``[bs, len_q, heads, P]``:
        ``[bs, len_q, heads * head_dim]``.
        """
        bs, len_q, heads, p, _ = sampling_locations.shape
        c = value.shape[1]
        h, w = hw
        grid = (2 * sampling_locations - 1).reshape(bs, len_q * heads * p, 1, 2)
        samples = F.grid_sample(
            value.view(bs, c, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).view(bs, c, len_q, heads, p)
        with torch.autocast(device_type=samples.device.type, enabled=False):
            x = samples.permute(0, 3, 2, 4, 1).reshape(bs, heads, len_q * p, c)
            x = torch.matmul(F.silu(torch.matmul(x, self.fine_lift)), self.fine_lift2)
            x = x.view(bs, heads, len_q, p, self.head_dim)
            weights = attention_weights.to(x.dtype).permute(0, 2, 1, 3).unsqueeze(-1)  # [bs, heads, len_q, P, 1]
            lifted = (x * weights).sum(3)  # [bs, heads, len_q, head_dim]
        return lifted.transpose(1, 2).reshape(bs, len_q, heads * self.head_dim)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value, value_spatial_shapes):
        """
        Args:
            query: ``[bs, query_length, C]``
            reference_points: ``[bs, query_length, 1, 4]`` normalized cxcywh boxes.
            value: the per-level values from ``TransformerDecoder.value_op``, the fine level first.
            value_spatial_shapes: the ``(h, w)`` of each level.

        Returns:
            ``[bs, query_length, C]``
        """
        bs, len_q = query.shape[:2]

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)
        logits = self.attention_weights(query).reshape(bs, len_q, self.num_heads, -1)

        num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
        wh = reference_points[:, :, None, :, 2:]
        offset = sampling_offsets * num_points_scale * wh * self.offset_scale
        sampling_locations = reference_points[:, :, None, :, :2] + offset

        attention_weights = F.softmax(logits, dim=-1)  # over the head's points, across the levels

        if self.fine_dim == 0:
            return ms_deformable_attention_core(
                value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
            )
        # the fine level's points come first and are read and lifted on their own; the other
        # levels go through the core
        p = self.num_points_list[0]
        output = self._read_fine(
            value[0], value_spatial_shapes[0], sampling_locations[:, :, :, :p], attention_weights[..., :p]
        )
        return output + ms_deformable_attention_core(
            value[1:],
            value_spatial_shapes[1:],
            sampling_locations[:, :, :, p:],
            attention_weights[..., p:],
            self.num_points_list[1:],
        )
