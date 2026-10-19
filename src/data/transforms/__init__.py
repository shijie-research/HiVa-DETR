"""
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

The evaluation transforms: a resize and the conversion of the PIL image to a float tensor.
"""

from typing import Any

import PIL.Image
import torch.nn as nn
import torchvision.transforms.v2 as T  # noqa: N812
import torchvision.transforms.v2.functional as F  # noqa: N812

from ...core import GLOBAL_CONFIG, create, register
from .._misc import Image

__all__ = ["Compose", "ConvertPILImage", "Resize"]

Resize = register()(T.Resize)


@register()
class ConvertPILImage(T.Transform):
    """PIL image -> tv_tensors.Image, as float in [0, 1] by default."""

    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.0
        return Image(inpt)


@register()
class Compose(T.Compose):
    """The transform pipeline of a dataset, built from the yaml's ``ops`` list of ``{type: <registered name>, ...}`` dicts."""

    def __init__(self, ops: list[dict | nn.Module]) -> None:
        super().__init__(transforms=[self._build(op) for op in ops])

    @staticmethod
    def _build(op: dict | nn.Module) -> nn.Module:
        if isinstance(op, nn.Module):
            return op
        args = {k: v for k, v in op.items() if k != "type"}
        return create(op["type"], GLOBAL_CONFIG, **args)
