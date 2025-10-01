"""
Deepfake Detection System
A comprehensive AI-powered system for detecting synthetic media and identity fraud
"""

__version__ = "1.0.0"
__author__ = "sam pautrat"
__email__ = "sampautrat101@gmail.com"

from src.detector import DeepfakeDetector
from src.preprocessing import PreprocessingPipeline

__all__ = [
    'DeepfakeDetector',
    'PreprocessingPipeline',
]
