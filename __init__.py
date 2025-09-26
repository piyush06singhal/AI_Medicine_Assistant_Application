"""
AI Medical Assistant
A comprehensive AI-powered medical assistant with NLP and Computer Vision capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Medical Assistant Team"
__email__ = "team@aimedicalassistant.com"

from .config import settings
from .utils.data_processing import DataProcessor
from .utils.model_utils import ModelManager

__all__ = [
    "settings",
    "DataProcessor", 
    "ModelManager"
]
