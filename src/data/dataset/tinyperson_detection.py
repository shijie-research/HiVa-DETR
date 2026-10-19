"""
HiVa-DETR

TinyPerson, read from the Hub (https://huggingface.co/datasets/shijli/tinyperson).
"""

import math

import torch

from ...core import register
from .hf_detection import HFDetection

__all__ = ["TinyPersonDetection"]

DENSE_FOLDER = "labeled_dense_images/"


@register()
class TinyPersonDetection(HFDetection):
    """
    The ``detection`` config of the TinyPerson repo: 794 train / 816 test images with the official
    annotations whole, read the way the TinyBenchmark tiny-person task reads them.

    - One class, ``person`` (label 0): sea and earth persons are not told apart.
    - The boxes that are neither ``ignore`` nor ``uncertain`` are the objects; the ``ignore`` and
      ``uncertain`` boxes are ignore regions (crowd boxes), kept in the ground truth, where
      ``TinyPersonEvaluator`` treats them as the benchmark does.
    - ``drop_dense`` leaves out the dense images (more than 220 boxes, 30 of the test split),
      which the benchmark does not score on: 786 test images remain.

    ``tile_size`` cuts every image into windows of that size (the last one in each direction
    flush with the image's edge, neighbours overlapping by at least ``tile_overlap`` pixels) at
    the image's own scale, which is the benchmark's way of testing on frames up to 3840 x 2160.
    An item is then one window: its image id is the window's index and its target the window's
    own size, so the predictions come back in window pixels, and ``TinyPersonEvaluator`` shifts
    them into the image and merges the windows. The window table rides along in the ground truth
    (``coco.dataset['tiles']``, ``[image id, x, y, w, h]`` per window), which is how the
    evaluator, built from the ground truth alone, finds it.
    """

    REPO = "shijli/tinyperson"
    CONFIG = "detection"
    CATEGORIES = [(0, "person")]

    def __init__(
        self,
        split,
        repo=None,
        config=None,
        transforms=None,
        revision=None,
        cache_dir=None,
        drop_dense=True,
        tile_size=None,
        tile_overlap=100,
    ):
        super().__init__(split, repo=repo, config=config, transforms=transforms, revision=revision, cache_dir=cache_dir)
        if drop_dense:
            keep = [i for i, name in enumerate(self.hf["file_name"]) if not name.startswith(DENSE_FOLDER)]
            self.hf = self.hf.select(keep)
        self.tiles = None
        if tile_size is not None:
            tile_w, tile_h = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
            self.tiles = self._tile_table(tile_w, tile_h, tile_overlap)

    def _tile_table(self, tile_w, tile_h, overlap):
        """``[image index, x, y, w, h]`` of every window, image by image, row-major within one."""
        tiles = []
        for index, row in enumerate(self.hf_meta.select_columns(["width", "height"])):
            w, h = row["width"], row["height"]
            for y in _window_starts(h, tile_h, overlap):
                for x in _window_starts(w, tile_w, overlap):
                    tiles.append([index, x, y, min(tile_w, w), min(tile_h, h)])
        return tiles

    def parse_objects(self, objects):
        n = len(objects["bbox"])
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # official xywh -> xyxy
        ignore = torch.tensor(objects["ignore"], dtype=torch.bool)
        uncertain = torch.tensor(objects["uncertain"], dtype=torch.bool)
        labels = torch.zeros((n,), dtype=torch.int64)
        return boxes, labels, (ignore | uncertain).to(torch.int64)

    def __len__(self):
        return len(self.tiles) if self.tiles is not None else len(self.hf)

    def load_item(self, index):
        if self.tiles is None:
            return super().load_item(index)
        image_index, x, y, w, h = self.tiles[index]
        row = self.hf[image_index]
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.crop((x, y, x + w, y + h))
        boxes, labels, iscrowd = self.parse_objects(row["objects"])
        boxes -= torch.tensor([x, y, x, y], dtype=boxes.dtype)
        return image, self._make_target(index, boxes, labels, iscrowd, w, h)

    @property
    def coco(self):
        if self._coco is None:
            self._coco = self.build_coco()
            if self.tiles is not None:
                self._coco.dataset["tiles"] = self.tiles
        return self._coco


def _window_starts(length, window, overlap):
    """The left (or top) edges of windows covering ``[0, length)``: evenly stepped, the last flush with the end."""
    if length <= window:
        return [0]
    count = math.ceil((length - overlap) / (window - overlap))
    step = (length - window) / (count - 1)
    return [round(i * step) for i in range(count)]
