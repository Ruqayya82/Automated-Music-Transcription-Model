"""
Audio Processing Module

Handles audio loading, preprocessing, and feature extraction
"""

from .audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor
from .pitch_detector import PitchDetector
from .onset_detector import OnsetDetector

__all__ = [
    'AudioLoader',
    'FeatureExtractor',
    'PitchDetector',
    'OnsetDetector'
]
