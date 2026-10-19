"""
HiVa-DETR
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.

The transformer building blocks: a plain MLP head, the post-norm encoder layer and stack used
for intra-scale attention, and the decoder layer with its gated residual.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .backbone.common import get_activation
from .deformable_attention import MSDeformableAttention

__all__ = ["MLP", "Gate", "TransformerDecoderLayer", "TransformerEncoder", "TransformerEncoderLayer"]


class MLP(nn.Module):
    """``num_layers`` linear layers with ``act`` between them and none after the last."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Self-attention then a feed-forward block, each with a residual and a LayerNorm after it."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        q = k = src if pos_embed is None else src + pos_embed
        residual = src
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)
        src = self.norm1(residual + self.dropout1(src))
        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(residual + self.dropout2(src))


class TransformerEncoder(nn.Module):
    """``num_layers`` copies of ``encoder_layer`` in sequence."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        return output


class Gate(nn.Module):
    """``norm(g1 * x1 + g2 * x2)`` with the two gates predicted from ``[x1, x2]``."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gates = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class TransformerDecoderLayer(nn.Module):
    """
    Self-attention over the queries, deformable cross-attention into the feature levels (merged
    through a Gate rather than add-and-norm), then a feed-forward block.
    """

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        fine_dim=0,
    ):
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, fine_dim=fine_dim)
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _self_attention(self, q, k, v, attn_mask):
        """
        ``self_attn`` (an ``nn.MultiheadAttention``, kept for its parameters) run through
        ``scaled_dot_product_attention`` with ``attn_mask`` ``[B, 1, Q, Q]`` (additive, ``-inf``
        where a query may not attend) broadcast over the heads.
        """
        mha = self.self_attn
        b, n, d = q.shape
        w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
        q = F.linear(q, w_q, b_q).view(b, n, mha.num_heads, -1).transpose(1, 2)
        k = F.linear(k, w_k, b_k).view(b, n, mha.num_heads, -1).transpose(1, 2)
        v = F.linear(v, w_v, b_v).view(b, n, mha.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return mha.out_proj(out.transpose(1, 2).reshape(b, n, d))

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2 = self._self_attention(q, k, target, attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        # cross attention
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        target = target + self.dropout4(target2)
        return self.norm3(target.clamp(min=-65504, max=65504))
