"""
Onset Detector Module

Detects note onsets (beginning of notes) in audio
"""

import librosa
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class OnsetDetector:
    """Detect note onsets in audio signals"""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize OnsetDetector
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Number of samples between frames
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def detect_onsets(self, audio: np.ndarray, 
                     units: str = 'time',
                     backtrack: bool = True) -> np.ndarray:
        """
        Detect note onsets in audio
        
        Args:
            audio: Audio signal
            units: 'time' for seconds, 'frames' for frame indices, 'samples' for sample indices
            backtrack: Backtrack detected onsets to previous local minimum of energy
            
        Returns:
            Array of onset times/frames/samples
        """
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=backtrack,
            units='frames'
        )
        
        # Convert to requested units
        if units == 'time':
            onsets = librosa.frames_to_time(onset_frames, sr=self.sample_rate, 
                                           hop_length=self.hop_length)
        elif units == 'samples':
            onsets = librosa.frames_to_samples(onset_frames, hop_length=self.hop_length)
        else:  # frames
            onsets = onset_frames
        
        logger.info(f"Detected {len(onsets)} onsets in {units}")
        return onsets
    
    def onset_strength(self, audio: np.ndarray) -> np.ndarray:
        """
        Calculate onset strength envelope
        
        Args:
            audio: Audio signal
            
        Returns:
            Onset strength envelope
        """
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        logger.info(f"Computed onset strength: {onset_env.shape}")
        return onset_env
    
    def detect_onsets_with_strength(self, audio: np.ndarray,
                                   threshold: float = 0.5,
                                   units: str = 'time') -> tuple:
        """
        Detect onsets and return onset strength envelope
        
        Args:
            audio: Audio signal
            threshold: Threshold for onset detection (0-1)
            units: 'time', 'frames', or 'samples'
            
        Returns:
            Tuple of (onsets, onset_strength_envelope)
        """
        # Get onset strength
        onset_env = self.onset_strength(audio)
        
        # Detect onsets with threshold
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
            units='frames'
        )
        
        # Convert to requested units
        if units == 'time':
            onsets = librosa.frames_to_time(onset_frames, sr=self.sample_rate,
                                           hop_length=self.hop_length)
        elif units == 'samples':
            onsets = librosa.frames_to_samples(onset_frames, hop_length=self.hop_length)
        else:
            onsets = onset_frames
        
        return onsets, onset_env
    
    def filter_close_onsets(self, onsets: np.ndarray, 
                           min_interval: float = 0.05) -> np.ndarray:
        """
        Filter out onsets that are too close together
        
        Args:
            onsets: Array of onset times (in seconds)
            min_interval: Minimum time interval between onsets (seconds)
            
        Returns:
            Filtered onsets
        """
        if len(onsets) == 0:
            return onsets
        
        filtered = [onsets[0]]
        for onset in onsets[1:]:
            if onset - filtered[-1] >= min_interval:
                filtered.append(onset)
        
        filtered = np.array(filtered)
        logger.info(f"Filtered onsets from {len(onsets)} to {len(filtered)}")
        return filtered
    
    def segment_by_onsets(self, audio: np.ndarray, 
                         onsets: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio by onset times
        
        Args:
            audio: Audio signal
            onsets: Onset times in seconds
            
        Returns:
            List of audio segments
        """
        # Convert onsets to samples
        onset_samples = librosa.time_to_samples(onsets, sr=self.sample_rate)
        
        # Add start and end points
        onset_samples = np.concatenate([[0], onset_samples, [len(audio)]])
        
        # Extract segments
        segments = []
        for i in range(len(onset_samples) - 1):
            start = onset_samples[i]
            end = onset_samples[i + 1]
            segments.append(audio[start:end])
        
        logger.info(f"Segmented audio into {len(segments)} segments")
        return segments
    
    def get_onset_intervals(self, onsets: np.ndarray, 
                           audio_duration: float) -> np.ndarray:
        """
        Get time intervals between consecutive onsets
        
        Args:
            onsets: Onset times in seconds
            audio_duration: Total duration of audio in seconds
            
        Returns:
            Array of intervals (durations) between onsets
        """
        if len(onsets) == 0:
            return np.array([audio_duration])
        
        # Add end time
        onset_with_end = np.concatenate([onsets, [audio_duration]])
        
        # Calculate intervals
        intervals = np.diff(onset_with_end)
        
        return intervals
    
    def detect_complex_onsets(self, audio: np.ndarray) -> dict:
        """
        Detect onsets using multiple methods and combine results
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with onset information from different methods
        """
        # Spectral flux
        onset_env_spectral = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            feature=librosa.feature.spectral_centroid
        )
        
        onsets_spectral = librosa.onset.onset_detect(
            onset_envelope=onset_env_spectral,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
            units='time'
        )
        
        # Energy-based
        onset_env_energy = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        onsets_energy = librosa.onset.onset_detect(
            onset_envelope=onset_env_energy,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
            units='time'
        )
        
        # Combine onsets (union of both methods)
        all_onsets = np.unique(np.concatenate([onsets_spectral, onsets_energy]))
        
        # Filter close onsets
        filtered_onsets = self.filter_close_onsets(all_onsets, min_interval=0.05)
        
        result = {
            'spectral_onsets': onsets_spectral,
            'energy_onsets': onsets_energy,
            'combined_onsets': all_onsets,
            'filtered_onsets': filtered_onsets,
            'spectral_strength': onset_env_spectral,
            'energy_strength': onset_env_energy
        }
        
        logger.info(f"Detected complex onsets: {len(filtered_onsets)} combined onsets")
        return result
    
    def align_onsets_to_beats(self, onsets: np.ndarray, 
                             tempo: float = 120.0,
                             time_signature: int = 4) -> np.ndarray:
        """
        Align onsets to beat grid (quantization)
        
        Args:
            onsets: Onset times in seconds
            tempo: Tempo in BPM
            time_signature: Number of beats per measure
            
        Returns:
            Quantized onset times
        """
        # Calculate beat duration
        beat_duration = 60.0 / tempo
        
        # Quantize onsets to nearest beat
        quantized_onsets = np.round(onsets / beat_duration) * beat_duration
        
        logger.info(f"Aligned {len(onsets)} onsets to beat grid (tempo={tempo})")
        return quantized_onsets
    
    def get_onset_statistics(self, onsets: np.ndarray, 
                            audio_duration: float) -> dict:
        """
        Calculate statistics about onsets
        
        Args:
            onsets: Onset times in seconds
            audio_duration: Total audio duration in seconds
            
        Returns:
            Dictionary of onset statistics
        """
        if len(onsets) == 0:
            return {
                'num_onsets': 0,
                'onset_rate': 0.0,
                'mean_interval': np.nan,
                'std_interval': np.nan,
                'min_interval': np.nan,
                'max_interval': np.nan
            }
        
        intervals = self.get_onset_intervals(onsets, audio_duration)
        
        stats = {
            'num_onsets': len(onsets),
            'onset_rate': len(onsets) / audio_duration,
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals)
        }
        
        logger.info(f"Onset statistics: {stats['num_onsets']} onsets, "
                   f"rate={stats['onset_rate']:.2f}/sec")
        return stats
