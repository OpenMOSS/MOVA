"""
DataConfig for VideoReward evaluation.

This module provides the DataConfig dataclass that holds data-related
configuration loaded from the model_config.json checkpoint file.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    meta_data: str = ""
    data_dir: str = ""
    meta_data_test: str = ""
    max_frame_pixels: int = 200704
    num_frames: Optional[int] = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: List[str] = field(default_factory=lambda: ["VQ", "MQ", "TA"])
    prompt_template_type: str = "detailed_special"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True
