"""rlm - Reinforcement Learning with Language Models.

A library for training and evaluating language models using
reinforcement learning techniques.
"""

__version__ = "0.1.0"
__author__ = "rlm contributors"

from rlm.trainer import RLMTrainer
from rlm.config import RLMConfig

__all__ = [
    "RLMTrainer",
    "RLMConfig",
    "__version__",
]
