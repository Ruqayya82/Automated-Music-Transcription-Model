"""
Pitch detection module for monophonic audio transcription.
"""

import librosa
import numpy as np
from typing import Tuple, Optional
import yaml


class PitchDetector:
    """Detect pitch (fundamental frequency) from audio signals."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize PitchDetector with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sr = self.config['audio']['sample_rate']
        self.fmin = self.config['pitch_detection']['fmin']
        self.fmax = self.config['pitch_detection']['fmax']
        self.frame_length = self.config['pitch_detection']['frame_length']
        self.hop_length = self.config['pitch_detection']['hop_length']
        self.threshold = self.config['pitch_detection']['threshold']
    
    def detect_pitch_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch using pYIN algorithm (probabilistic YIN).
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (pitch_values, voiced_flags)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Apply threshold to confidence
        voiced_flag = voiced_probs > self.threshold
        
        return f0, voiced_flag
    
    def detect_pitch_yin(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect pitch using YIN algorithm.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pitch values in Hz
        """
        f0 = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        return f0
    
    def detect_pitch_autocorrelation(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect pitch using autocorrelation method.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pitch values in Hz
        """
        # Compute autocorrelation
        autocorr = librosa.autocorrelate(audio, max_size=self.frame_length)
        
        # Find peaks in autocorrelation
        # This is a simplified implementation
        # In practice, you'd want more sophisticated peak detection
        peaks = librosa.util.peak_pick(
            autocorr,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.5,
            wait=10
        )
        
        if len(peaks) > 0:
            # First peak after zero lag corresponds to fundamental period
            period = peaks[0]
            f0 = self.sr / period
        else:
            f0 = 0
        
        return np.array([f0])
    
    def hz_to_midi(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Convert frequency in Hz to MIDI note number.
        
        Args:
            frequencies: Frequency values in Hz
            
        Returns:
            MIDI note numbers
        """
        # Handle NaN and zero values
        valid_mask = np.isfinite(frequencies) & (frequencies > 0)
        midi_notes = np.zeros_like(frequencies)
        
        # Convert valid frequencies to MIDI
        midi_notes[valid_mask] = librosa.hz_to_midi(frequencies[valid_mask])
        
        return midi_notes
    
    def midi_to_note_name(self, midi_note: float) -> str:
        """
        Convert MIDI note number to note name.
        
        Args:
            midi_note: MIDI note number
            
        Returns:
            Note name (e.g., 'C4', 'A#5')
        """
        if not np.isfinite(midi_note) or midi_note <= 0:
            return 'Rest'
        
        return librosa.midi_to_note(int(round(midi_note)))
    
    def quantize_pitch_to_midi(self, frequencies: np.ndarray, 
                                voiced_flags: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Quantize continuous pitch values to discrete MIDI notes.
        
        Args:
            frequencies: Pitch values in Hz
            voiced_flags: Boolean array indicating voiced segments
            
        Returns:
            Quantized MIDI note numbers
        """
        # Convert to MIDI
        midi_notes = self.hz_to_midi(frequencies)
        
        # Round to nearest semitone
        midi_notes = np.round(midi_notes)
        
        # Apply voiced flags if provided
        if voiced_flags is not None:
            midi_notes[~voiced_flags] = 0
        
        return midi_notes
    
    def smooth_pitch_contour(self, pitches: np.ndarray, 
                             window_size: int = 5) -> np.ndarray:
        """
        Smooth pitch contour using median filtering.
        
        Args:
            pitches: Raw pitch values
            window_size: Size of median filter window
            
        Returns:
            Smoothed pitch values
        """
        from scipy.ndimage import median_filter
        
        # Only smooth valid (non-zero) values
        valid_mask = pitches > 0
        smoothed = pitches.copy()
        
        if np.any(valid_mask):
            smoothed[valid_mask] = median_filter(
                pitches[valid_mask], 
                size=window_size, 
                mode='nearest'
            )
        
        return smoothed
    
    def detect_and_quantize(self, audio: np.ndarray, 
                           smooth: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pitch detection and quantization pipeline.
        
        Args:
            audio: Input audio signal
            smooth: Whether to smooth pitch contour
            
        Returns:
            Tuple of (frequencies, voiced_flags, midi_notes)
        """
        # Detect pitch
        frequencies, voiced_flags = self.detect_pitch_pyin(audio)
        
        # Smooth if requested
        if smooth:
            frequencies = self.smooth_pitch_contour(frequencies)
        
        # Quantize to MIDI
        midi_notes = self.quantize_pitch_to_midi(frequencies, voiced_flags)
        
        return frequencies, voiced_flags, midi_notes
    
    def get_pitch_times(self, audio: np.ndarray) -> np.ndarray:
        """
        Get time values corresponding to pitch estimates.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Time values in seconds
        """
        n_frames = 1 + int((len(audio) - self.frame_length) / self.hop_length)
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sr,
            hop_length=self.hop_length
        )
        return times
