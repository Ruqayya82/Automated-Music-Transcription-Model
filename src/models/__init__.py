"""
Machine Learning Models Module

Contains model architectures for music transcription
"""

from .transcription_model import TranscriptionModel
from .pitch_onset_cnn import PitchOnsetCNN
from .model_trainer import ModelTrainer

__all__ = [
    'TranscriptionModel',
    'PitchOnsetCNN',
    'ModelTrainer'
]
