"""
Audio Loader Module

Handles loading and basic preprocessing of audio files
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load and preprocess audio files"""
    
    def __init__(self, sample_rate: int = 22050, mono: bool = True):
        """
        Initialize AudioLoader
        
        Args:
            sample_rate: Target sample rate for audio
            mono: Convert to mono if True
        """
        self.sample_rate = sample_rate
        self.mono = mono
        
    def load_audio(self, file_path: str, duration: Optional[float] = None,
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            duration: Duration to load (seconds), None for entire file
            offset: Start time (seconds)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            logger.info(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio: Audio data
            top_db: Threshold for silence detection
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        logger.info(f"Trimmed audio from {len(audio)} to {len(trimmed)} samples")
        return trimmed
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, 
                      target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        logger.info(f"Resampled from {orig_sr} Hz to {target_sr} Hz")
        return resampled
    
    def save_audio(self, audio: np.ndarray, file_path: str, 
                   sample_rate: Optional[int] = None):
        """
        Save audio to file
        
        Args:
            audio: Audio data to save
            file_path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        sr = sample_rate or self.sample_rate
        try:
            sf.write(file_path, audio, sr)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {str(e)}")
            raise
    
    def get_duration(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> float:
        """
        Get duration of audio in seconds
        
        Args:
            audio: Audio data
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Duration in seconds
        """
        sr = sample_rate or self.sample_rate
        return len(audio) / sr
    
    def preprocess(self, file_path: str, trim: bool = True, 
                  normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
            trim: Trim silence if True
            normalize: Normalize audio if True
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        audio, sr = self.load_audio(file_path)
        
        if trim:
            audio = self.trim_silence(audio)
        
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio, sr
