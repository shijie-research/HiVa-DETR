"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from . import yaml_utils
from .workspace import GLOBAL_CONFIG, create, register
from .yaml_config import YAMLConfig
from .yaml_utils import load_config, merge_config, merge_dict, parse_cli
