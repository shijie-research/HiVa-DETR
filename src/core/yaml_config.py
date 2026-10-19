"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy

import torch.nn as nn

from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict

__all__ = ["YAMLConfig"]


class YAMLConfig:
    """
    The components of an evaluation, built lazily from a yaml file: the model, the
    postprocessor, the validation loader and the evaluator. Each is created from the registry
    the first time it is read, using the merged global config.
    """

    def __init__(self, cfg_path: str, **kwargs) -> None:
        cfg = load_config(cfg_path)
        self.yaml_cfg = merge_dict(cfg, kwargs)
        self._model = self._postprocessor = self._val_dataloader = self._evaluator = None

    @property
    def global_cfg(self) -> dict:
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return self._model

    @property
    def postprocessor(self) -> nn.Module:
        if self._postprocessor is None:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return self._postprocessor

    @property
    def val_dataloader(self):
        if self._val_dataloader is None:
            global_cfg = self.global_cfg
            batch_size = global_cfg["val_dataloader"].pop("total_batch_size")
            self._val_dataloader = create("val_dataloader", global_cfg, batch_size=batch_size)
        return self._val_dataloader

    @property
    def evaluator(self):
        if self._evaluator is None:
            from ..data import get_coco_api_from_dataset

            coco_gt = get_coco_api_from_dataset(self.val_dataloader.dataset)
            self._evaluator = create("evaluator", self.global_cfg, coco_gt=coco_gt)
        return self._evaluator

    def __repr__(self):
        return repr(copy.deepcopy(self.yaml_cfg))
