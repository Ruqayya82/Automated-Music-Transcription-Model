"""
Feature extraction from audio signals.
"""

import librosa
import numpy as np
from typing import Tuple, Dict
import yaml


class FeatureExtractor:
    """Extract spectral and temporal features from audio."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize FeatureExtractor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sr = self.config['audio']['sample_rate']
        self.n_fft = self.config['audio']['n_fft']
        self.hop_length = self.config['audio']['hop_length']
        self.n_mels = self.config['audio']['n_mels']
        self.fmin = self.config['audio']['fmin']
        self.fmax = self.config['audio']['fmax']
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Complex-valued STFT matrix
        """
        stft = librosa.stft(audio, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length)
        return stft
    
    def compute_spectrogram(self, audio: np.ndarray, 
                           to_db: bool = True) -> np.ndarray:
        """
        Compute magnitude spectrogram.
        
        Args:
            audio: Input audio signal
            to_db: Convert to decibels if True
            
        Returns:
            Magnitude spectrogram
        """
        stft = self.compute_stft(audio)
        mag_spec = np.abs(stft)
        
        if to_db:
            mag_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)
        
        return mag_spec
    
    def compute_mel_spectrogram(self, audio: np.ndarray, 
                               to_db: bool = True) -> np.ndarray:
        """
        Compute mel-scale spectrogram.
        
        Args:
            audio: Input audio signal
            to_db: Convert to decibels if True
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def compute_cqt(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Constant-Q Transform (good for music analysis).
        
        Args:
            audio: Input audio signal
            
        Returns:
            CQT matrix
        """
        cqt = librosa.cqt(audio, 
                         sr=self.sr, 
                         hop_length=self.hop_length,
                         fmin=self.fmin)
        
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db
    
    def compute_chromagram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chromagram (pitch class representation).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Chromagram
        """
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin
        )
        return chroma
    
    def compute_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral centroid (brightness measure).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Spectral centroid over time
        """
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return centroid[0]
    
    def compute_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute RMS energy over time.
        
        Args:
            audio: Input audio signal
            
        Returns:
            RMS energy
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return rms[0]
    
    def compute_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute zero crossing rate.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return zcr[0]
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features for comprehensive analysis.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of feature arrays
        """
        features = {
            'spectrogram': self.compute_spectrogram(audio),
            'mel_spectrogram': self.compute_mel_spectrogram(audio),
            'cqt': self.compute_cqt(audio),
            'chromagram': self.compute_chromagram(audio),
            'spectral_centroid': self.compute_spectral_centroid(audio),
            'rms_energy': self.compute_rms_energy(audio),
            'zero_crossing_rate': self.compute_zero_crossing_rate(audio)
        }
        
        return features
    
    def frames_to_time(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert frame indices to time in seconds.
        
        Args:
            frames: Frame indices
            
        Returns:
            Time values in seconds
        """
        return librosa.frames_to_time(frames, 
                                     sr=self.sr, 
                                     hop_length=self.hop_length)
    
    def time_to_frames(self, times: np.ndarray) -> np.ndarray:
        """
        Convert time in seconds to frame indices.
        
        Args:
            times: Time values in seconds
            
        Returns:
            Frame indices
        """
        return librosa.time_to_frames(times, 
                                     sr=self.sr, 
                                     hop_length=self.hop_length)
