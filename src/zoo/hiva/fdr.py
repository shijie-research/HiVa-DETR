"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Fine-grained Distribution Refinement (FDR), D-FINE's box regression: each box edge is predicted
as a distribution over ``reg_max + 1`` bins whose values are the non-uniform weighting function
W(n); the edge offset is the expectation under that distribution. This module holds the
weighting function, the decoding of boxes from edge distances, the Integral layer that takes the
expectation, and the LQE head that turns the distribution's sharpness into a quality score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...nn.transformer import MLP

__all__ = ["LQE", "Integral", "distance2bbox", "weighting_function"]


def weighting_function(reg_max, up, reg_scale):
    """
    The non-uniform weighting function W(n): ``reg_max + 1`` values from ``-2 * up * reg_scale``
    to ``2 * up * reg_scale``, zero in the middle, spaced geometrically so that bins are fine near
    zero and coarse at the ends.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = upper_bound1 * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [step**i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def distance2bbox(points, distance, reg_scale):
    """
    Boxes from reference boxes and predicted edge distances.

    Args:
        points (Tensor): ``(..., 4)`` reference boxes as [cx, cy, w, h].
        distance (Tensor): ``(..., 4)`` distances of the left, top, right and bottom edges from
            the reference box's edges, outwards positive, in units of the reference size /
            reg_scale.
        reg_scale: curvature of W(n).

    Returns:
        Tensor: ``(..., 4)`` boxes as [cx, cy, w, h].
    """
    unit = points[..., 2:] / abs(reg_scale)
    half = 0.5 * points[..., 2:]
    lt = points[..., :2] - half - distance[..., :2] * unit
    rb = points[..., :2] + half + distance[..., 2:] * unit
    return torch.cat([(lt + rb) / 2, rb - lt], -1)  # xyxy -> cxcywh


class Integral(nn.Module):
    """
    The expectation of a binned edge distribution: ``sum_n softmax(x)[n] * W(n)`` for each of
    the four edges, where ``project`` holds W(n) (see ``weighting_function``).
    """

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """
    Location quality estimator: adds to the class scores a term predicted from the top-``k``
    probabilities (and their mean) of each edge distribution, so sharper distributions score
    higher.
    """

    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)

    def forward(self, scores, pred_corners):
        b, n, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(b, n, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1).reshape(b, n, -1)
        return scores + self.reg_conf(stat)
