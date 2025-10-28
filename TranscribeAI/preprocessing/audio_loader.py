"""
Audio loading and preprocessing utilities.
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import yaml


class AudioLoader:
    """Load and preprocess audio files for transcription."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize AudioLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sr = self.config['audio']['sample_rate']
        self.frame_length = self.config['audio']['frame_length']
        self.hop_length = self.config['audio']['hop_length']
    
    def load_audio(self, audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (uses config default if None)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
            print(f"Loaded audio: {audio_path}")
            print(f"Duration: {len(audio)/sample_rate:.2f} seconds")
            print(f"Sample rate: {sample_rate} Hz")
            
            return audio, sample_rate
        
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        # Peak normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """
        Trim leading and trailing silence from audio.
        
        Args:
            audio: Input audio array
            top_db: Threshold (in dB) below reference to consider as silence
            
        Returns:
            Trimmed audio array
        """
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio
    
    def preprocess(self, audio_path: str, 
                   normalize: bool = True, 
                   trim: bool = True) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline for audio.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio
            trim: Whether to trim silence
            
        Returns:
            Tuple of (preprocessed_audio, sample_rate)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Trim silence
        if trim:
            audio = self.trim_silence(audio)
        
        # Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: Optional[int] = None):
        """
        Save audio array to file.
        
        Args:
            audio: Audio array to save
            output_path: Output file path
            sr: Sample rate (uses config default if None)
        """
        if sr is None:
            sr = self.sr
        
        sf.write(output_path, audio, sr)
        print(f"Audio saved to: {output_path}")
