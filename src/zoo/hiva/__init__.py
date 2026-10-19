"""
HiVa-DETR: ``HiVaDETR`` wires the backbone, the ``HybridEncoder`` with the stride-2 value level
and the ``DFINETransformer`` decoder that samples it; ``PostProcessor`` turns its outputs into
detections. Importing the package registers all of them for the configs.
"""

from .decoder import DFINETransformer
from .hiva_detr import HiVaDETR
from .hybrid_encoder import HybridEncoder
from .postprocessor import PostProcessor

__all__ = ["DFINETransformer", "HiVaDETR", "HybridEncoder", "PostProcessor"]
