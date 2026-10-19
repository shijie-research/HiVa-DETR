"""
HiVa-DETR

Datasets are read straight from the Hugging Face Hub: one ``HFDetection`` subclass per repo,
and one evaluator per dataset protocol.
"""

from .aitod_detection import AITOD_CLASSES, AITODDetection
from .aitod_eval import AITODEvaluator
from .hf_detection import HFDetection, get_coco_api_from_dataset
from .tinyperson_detection import TinyPersonDetection
from .tinyperson_eval import TinyPersonEvaluator
from .visdrone_detection import VISDRONE_CLASSES, VisDroneDetection
from .visdrone_eval import VisDroneEvaluator

__all__ = [
    "AITOD_CLASSES",
    "AITODDetection",
    "AITODEvaluator",
    "HFDetection",
    "TinyPersonDetection",
    "TinyPersonEvaluator",
    "VISDRONE_CLASSES",
    "VisDroneDetection",
    "VisDroneEvaluator",
    "get_coco_api_from_dataset",
]
