"""
HiVa-DETR
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The D-FINE decoder with HiVa's value level: a deformable-attention decoder stack with
Fine-grained Distribution Refinement, fed by the best encoder tokens, whose cross-attention also
samples the encoder's stride-2 map as values. This is the inference path only.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...core import register
from ...nn.functional import inverse_sigmoid
from ...nn.transformer import MLP, TransformerDecoderLayer
from .fdr import LQE, Integral, distance2bbox, weighting_function

__all__ = ["DFINETransformer", "TransformerDecoder"]


class TransformerDecoder(nn.Module):
    """
    The decoder stack with Fine-grained Distribution Refinement: every layer predicts a
    correction to the binned edge distributions of the layer before, the boxes are decoded from
    the accumulated distribution around the first layer's reference boxes, and a location
    quality estimator adjusts the class scores of the last layer.
    """

    def __init__(self, decoder_layer, num_layers, num_head, reg_max, fine_dim=0):
        super().__init__()
        self.num_layers = num_layers
        self.num_head = num_head
        self.reg_max = reg_max
        self.fine_dim = fine_dim  # > 0: the first level is the raw value level, so many channels wide
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])

    def value_op(self, feats):
        """
        The levels ``[bs, C, h, w]`` as the cross-attention's values: the encoder levels split
        across the heads, ``[bs, heads, c, h * w]``, the value level whole, ``[bs, c, h * w]``.
        """
        values = []
        for level, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            value = feat.flatten(2).float()
            if level == 0 and self.fine_dim > 0:
                values.append(value)
            else:
                values.append(value.reshape(bs, self.num_head, -1, h * w))
        return values

    def forward(
        self,
        target,
        ref_points_unact,
        feats,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(feats)
        project = weighting_function(self.reg_max, up, reg_scale)
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # the first layer predicts plain boxes; they anchor the distributions of every layer
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                ref_points_initial = pre_bboxes.detach()

            # refine the edge distributions, carrying the previous layer's correction along
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if i == self.num_layers - 1:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                return inter_ref_bbox, scores

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()


@register()
class DFINETransformer(nn.Module):
    """
    Args:
        feat_channels / feat_strides / num_levels: the encoder levels.
        num_layers: decoder depth.
        reg_max / reg_scale: the FDR edge distributions.
        num_queries: the number of encoder tokens taken as initial queries.
        anchor_cells: the anchor box of every level's tokens is so many cells of that level,
            square in pixels.
        fine_channels: with a value above 0, the encoder's ``fine`` map (HiVa's stride-2 value
            level, so many channels wide) is read by every decoder layer's deformable
            cross-attention as its first, finest level (``MSDeformableAttention``'s
            ``fine_dim``). It is values only: no token of it enters the query selection, gets an
            anchor or is scored. ``num_points`` then has one more entry, its first for this level.
        query_budget: how many encoder tokens become queries. ``fixed``: ``num_queries`` for
            every image. ``threshold``: the number of valid tokens whose best class score exceeds
            ``count_threshold`` plus ``count_margin``, rounded up to a multiple of
            ``count_round`` and clamped to ``count_range``. The images of a batch are padded to
            the largest budget; the padded queries are masked out of the attention and the
            detections.
    """

    __share__ = ["num_classes"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        feat_channels=(256, 256, 256, 256),
        feat_strides=(4, 8, 16, 32),
        num_levels=4,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        eps=1e-2,
        reg_max=32,
        reg_scale=4.0,
        num_queries=300,
        anchor_cells=3,
        fine_channels=0,
        query_budget="fixed",
        count_threshold=0.15,
        count_margin=200,
        count_round=100,
        count_range=(300, 1500),
    ):
        super().__init__()
        assert query_budget in ("fixed", "threshold"), query_budget
        assert len(feat_channels) == num_levels == len(feat_strides)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = list(feat_strides)
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eps = eps
        self.anchor_cells = anchor_cells
        self.num_layers = num_layers
        self.reg_max = reg_max
        self.num_queries = num_queries
        self.query_budget = query_budget
        self.count_threshold = count_threshold
        self.count_margin = count_margin
        self.count_round = count_round
        self.count_range = tuple(count_range)
        self.fine_channels = fine_channels
        num_value_levels = num_levels + (1 if fine_channels > 0 else 0)
        if isinstance(num_points, (list, tuple)):
            assert len(num_points) == num_value_levels, (
                f"num_points needs one entry per cross-attention level ({num_value_levels}, "
                f"the value level first), got {list(num_points)}"
            )
            num_points = list(num_points)

        # backbone feature projection
        self.input_proj = nn.ModuleList(self._proj(c) for c in feat_channels)

        # transformer
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_value_levels,
            n_points=num_points,
            fine_dim=fine_channels,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers, nhead, reg_max, fine_dim=fine_channels)

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # encoder output: token scores and boxes for query selection
        self.enc_output = nn.Sequential(
            OrderedDict([("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim))])
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder heads, one per layer
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3) for _ in range(num_layers)]
        )
        self.integral = Integral(self.reg_max)

    def _proj(self, in_channels):
        """A conv-BN projection to ``hidden_dim`` (nothing, for a level already ``hidden_dim`` wide)."""
        if in_channels == self.hidden_dim:
            return nn.Identity()
        conv = nn.Conv2d(in_channels, self.hidden_dim, 1, 1, bias=False)
        return nn.Sequential(OrderedDict([("conv", conv), ("norm", nn.BatchNorm2d(self.hidden_dim))]))

    def _generate_anchors(self, spatial_shapes, dtype=torch.float32, device="cpu"):
        """
        One anchor box (as logits) per token of every level, ``anchor_cells`` cells of the level
        wide; tokens whose centre is too close to the border are marked invalid.
        """
        anchors = []
        for h, w in spatial_shapes:
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * self.anchor_cells / torch.tensor([w, h], dtype=dtype)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        centres = anchors[..., :2]
        valid_mask = ((centres > self.eps) * (centres < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _budgets(self, scores) -> list[int]:
        """
        Per image, the number of queries under ``query_budget='threshold'``: the count of tokens
        above the threshold plus ``count_margin``, rounded up to ``count_round`` and clamped to
        ``count_range``.
        """
        count = (scores > self.count_threshold).sum(1)
        lo, hi = self.count_range
        budget = torch.div(count + self.count_margin + self.count_round - 1, self.count_round, rounding_mode="floor")
        return (budget * self.count_round).clamp(lo, hi).tolist()

    def _get_decoder_input(self, memory, spatial_shapes):
        """
        The initial queries: the best encoder tokens, ``num_queries`` of them or the image's
        budget (``query_budget``). Returns their contents and boxes (as logits), padded to the
        largest count in the batch, and the real query count of every image.
        """
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        b = memory.shape[0]
        if b > 1:
            anchors = anchors.repeat(b, 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        if self.query_budget == "fixed":
            batch_queries_num = [self.num_queries] * b
        else:
            scores = F.sigmoid(enc_outputs_logits.max(-1).values)
            scores = scores.masked_fill(~valid_mask[..., 0].expand(b, -1), 0.0)
            batch_queries_num = self._budgets(scores)
        batch_queries_num = [min(n, enc_outputs_logits.shape[1]) for n in batch_queries_num]
        k = max(batch_queries_num)

        _, topk_ind = torch.topk(enc_outputs_logits.max(-1).values, k, dim=-1)

        def gather(x):
            return x.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        topk_memory, topk_anchors = gather(output_memory), gather(anchors)
        topk_bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors
        if min(batch_queries_num) < k:
            # the images with a smaller budget are padded: their surplus tokens are zeroed here and
            # masked everywhere else by batch_queries_num
            counts = torch.tensor(batch_queries_num, device=memory.device)
            pad = (torch.arange(k, device=memory.device)[None, :] >= counts[:, None])[..., None]
            topk_memory = topk_memory.masked_fill(pad, 0.0)
            topk_bbox_unact = topk_bbox_unact.masked_fill(pad, 0.0)
        return topk_memory.detach(), topk_bbox_unact.detach(), batch_queries_num

    @staticmethod
    def _padding_attn_mask(batch_queries_num, device):
        """
        A ``[B, 1, Q, Q]`` additive self-attention mask hiding every image's padded queries from
        its real ones and vice versa (padding still sees itself), or ``None`` when no image is
        padded.
        """
        num_queries = max(batch_queries_num)
        if min(batch_queries_num) == num_queries:
            return None
        counts = torch.tensor(batch_queries_num, device=device)
        real = torch.arange(num_queries, device=device)[None, :] < counts[:, None]  # [B, Q]
        blocked = (real[:, :, None] != real[:, None, :])[:, None]
        return torch.zeros(blocked.shape, dtype=torch.float32, device=device).masked_fill(blocked, -torch.inf)

    def forward(self, encoder_out):
        feats = encoder_out["feats"]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        spatial_shapes = [list(feat.shape[2:]) for feat in proj_feats]
        memory = torch.concat([feat.flatten(2).permute(0, 2, 1) for feat in proj_feats], 1)

        # the cross-attention's levels: the value level first; the selection and the anchors stay
        # on the encoder levels
        value_feats, value_shapes = proj_feats, spatial_shapes
        if self.fine_channels > 0:
            fine = encoder_out["fine"]
            value_feats, value_shapes = [fine, *proj_feats], [list(fine.shape[2:]), *spatial_shapes]

        contents, boxes_unact, batch_queries_num = self._get_decoder_input(memory, spatial_shapes)
        attn_mask = self._padding_attn_mask(batch_queries_num, memory.device)
        out_bboxes, out_logits = self.decoder(
            contents,
            boxes_unact,
            value_feats,
            value_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
        )

        num_queries = max(batch_queries_num)
        if min(batch_queries_num) < num_queries:
            # padded queries never become detections
            counts = torch.tensor(batch_queries_num, device=memory.device)
            pad = torch.arange(num_queries, device=memory.device)[None, :] >= counts[:, None]
            out_logits = out_logits.masked_fill(pad[:, :, None], -1e4)

        return {"pred_logits": out_logits, "pred_boxes": out_bboxes, "batch_queries_num": batch_queries_num}
