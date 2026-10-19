"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision

from ...core import register

__all__ = ["PostProcessor"]


@register()
class PostProcessor(nn.Module):
    """
    Decoder outputs -> per-image detections in original-image pixels.

    ``forward`` takes the decoder's ``pred_logits`` [B, Q, C] and ``pred_boxes`` [B, Q, 4]
    (normalized cxcywh) and the (w, h) each image's predictions are to be scaled by, and returns
    one ``{"labels", "boxes", "scores"}`` dict per image, boxes as xyxy pixels. Every (query,
    class) pair is a candidate and the top ones are kept, so a query can yield more than one
    detection (as D-FINE does). An image keeps at most ``min(num_top_queries, Q)`` detections,
    ``Q`` its own number of real queries.
    """

    __share__ = ["num_classes"]

    def __init__(self, num_classes=80, num_top_queries=1500) -> None:
        super().__init__()
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        k = min(self.num_top_queries, logits.shape[1])

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred = bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        scores, index = torch.topk(F.sigmoid(logits).flatten(1), k, dim=-1)
        labels = index % self.num_classes
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, 4))

        keep = [min(k, int(n)) for n in outputs["batch_queries_num"]]
        return [
            dict(labels=lab[:n], boxes=box[:n], scores=sco[:n]) for lab, box, sco, n in zip(labels, boxes, scores, keep)
        ]
