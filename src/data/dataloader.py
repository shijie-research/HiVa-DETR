"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as data

from ..core import register

__all__ = ["BatchImageCollateFunction", "DataLoader"]


@register()
class DataLoader(data.DataLoader):
    """torch's DataLoader, registered for the configs."""

    __inject__ = ["dataset", "collate_fn"]


@register()
class BatchImageCollateFunction:
    """
    Stack the images of a batch. With ``pad_to_multiple``, images of different sizes are
    zero-padded on the bottom and right to the largest height and width in the batch, rounded up
    to that multiple. Each target then records ``resized_size`` (its image before padding) and
    ``padded_size`` (after), both as (w, h), which map the predictions back to the original image.
    """

    def __init__(self, pad_to_multiple=None) -> None:
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, items):
        targets = [x[1] for x in items]
        if self.pad_to_multiple:
            images = self._pad_to_common_size([x[0] for x in items], targets)
        else:
            images = torch.cat([x[0][None] for x in items], dim=0)
        return images, targets

    def _pad_to_common_size(self, images, targets):
        m = self.pad_to_multiple
        pad_h = -(-max(im.shape[-2] for im in images) // m) * m
        pad_w = -(-max(im.shape[-1] for im in images) // m) * m
        padded = []
        for im, tg in zip(images, targets):
            h, w = im.shape[-2:]
            padded.append(F.pad(im, (0, pad_w - w, 0, pad_h - h)))
            tg["resized_size"] = torch.tensor([w, h])
            tg["padded_size"] = torch.tensor([pad_w, pad_h])
        return torch.stack(padded, dim=0)
