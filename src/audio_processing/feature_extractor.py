"""
Feature Extractor Module

Extracts audio features for machine learning models
"""

import librosa
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from audio signals"""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048,
                 hop_length: int = 512, n_mels: int = 128):
        """
        Initialize FeatureExtractor
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def extract_mel_spectrogram(self, audio: np.ndarray, 
                               fmin: float = 30.0,
                               fmax: float = 8000.0) -> np.ndarray:
        """
        Extract mel spectrogram
        
        Args:
            audio: Audio signal
            fmin: Minimum frequency
            fmax: Maximum frequency
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=fmin,
            fmax=fmax
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        logger.info(f"Extracted mel spectrogram: {mel_spec_db.shape}")
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio signal
            n_mfcc: Number of MFCCs to extract
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        logger.info(f"Extracted MFCC: {mfcc.shape}")
        return mfcc
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chromagram features
        
        Args:
            audio: Audio signal
            
        Returns:
            Chromagram
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        logger.info(f"Extracted chroma: {chroma.shape}")
        return chroma
    
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral contrast
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral contrast
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        logger.info(f"Extracted spectral contrast: {contrast.shape}")
        return contrast
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract zero crossing rate
        
        Args:
            audio: Audio signal
            
        Returns:
            Zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        logger.info(f"Extracted ZCR: {zcr.shape}")
        return zcr
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral centroid
        """
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        logger.info(f"Extracted spectral centroid: {centroid.shape}")
        return centroid
    
    def extract_cqt(self, audio: np.ndarray, n_bins: int = 84,
                    bins_per_octave: int = 12) -> np.ndarray:
        """
        Extract Constant-Q Transform (CQT) - useful for pitch detection
        
        Args:
            audio: Audio signal
            n_bins: Number of frequency bins
            bins_per_octave: Number of bins per octave
            
        Returns:
            CQT spectrogram
        """
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )
        
        # Convert to dB scale
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        logger.info(f"Extracted CQT: {cqt_db.shape}")
        return cqt_db
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of features
        """
        features = {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
            'chroma': self.extract_chroma(audio),
            'spectral_contrast': self.extract_spectral_contrast(audio),
            'zcr': self.extract_zero_crossing_rate(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'cqt': self.extract_cqt(audio)
        }
        logger.info(f"Extracted all features")
        return features
    
    def normalize_features(self, features: np.ndarray, 
                          axis: Optional[int] = None) -> np.ndarray:
        """
        Normalize features using standardization
        
        Args:
            features: Feature array
            axis: Axis along which to normalize
            
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=axis, keepdims=True)
        std = np.std(features, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized = (features - mean) / std
        return normalized
    
    def frames_to_time(self, frames: int) -> float:
        """
        Convert frame index to time in seconds
        
        Args:
            frames: Frame index
            
        Returns:
            Time in seconds
        """
        return librosa.frames_to_time(frames, sr=self.sample_rate, 
                                     hop_length=self.hop_length)
    
    def time_to_frames(self, time: float) -> int:
        """
        Convert time to frame index
        
        Args:
            time: Time in seconds
            
        Returns:
            Frame index
        """
        return librosa.time_to_frames(time, sr=self.sample_rate,
                                     hop_length=self.hop_length)
