"""
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.utils.data as data


class DetDataset(data.Dataset):
    """A detection dataset is ``load_item``, the raw ``(image, target)`` of one index, plus a transform pipeline."""

    transforms = None

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def load_item(self, index):
        raise NotImplementedError
