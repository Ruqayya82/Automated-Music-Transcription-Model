"""
Pitch Detector Module

Detects pitch/frequency from monophonic audio
"""

import librosa
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PitchDetector:
    """Detect pitch from audio signals"""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512,
                 fmin: float = 65.0, fmax: float = 2093.0):
        """
        Initialize PitchDetector
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Number of samples between frames
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        
    def detect_pitch_pyin(self, audio: np.ndarray, 
                         frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pitch using pYIN algorithm (probabilistic YIN)
        Best for monophonic audio
        
        Args:
            audio: Audio signal
            frame_length: Length of analysis frame
            
        Returns:
            Tuple of (f0, voiced_flag, voiced_probs)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            frame_length=frame_length,
            hop_length=self.hop_length
        )
        logger.info(f"Detected pitch using pYIN: {f0.shape}")
        return f0, voiced_flag, voiced_probs
    
    def detect_pitch_yin(self, audio: np.ndarray,
                        frame_length: int = 2048) -> np.ndarray:
        """
        Detect pitch using YIN algorithm
        
        Args:
            audio: Audio signal
            frame_length: Length of analysis frame
            
        Returns:
            Fundamental frequency estimates
        """
        # Using pyin but only returning f0
        f0, _, _ = self.detect_pitch_pyin(audio, frame_length)
        return f0
    
    def frequency_to_midi(self, frequency: np.ndarray) -> np.ndarray:
        """
        Convert frequency (Hz) to MIDI note number
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            MIDI note numbers
        """
        # Handle NaN values
        valid_mask = ~np.isnan(frequency)
        midi = np.full_like(frequency, np.nan)
        
        if np.any(valid_mask):
            midi[valid_mask] = librosa.hz_to_midi(frequency[valid_mask])
        
        return midi
    
    def midi_to_note_name(self, midi_note: float) -> str:
        """
        Convert MIDI note number to note name
        
        Args:
            midi_note: MIDI note number
            
        Returns:
            Note name (e.g., 'C4', 'A#5')
        """
        if np.isnan(midi_note):
            return "Rest"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                     'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = int(midi_note // 12) - 1
        note_idx = int(midi_note % 12)
        return f"{note_names[note_idx]}{octave}"
    
    def smooth_pitch_contour(self, f0: np.ndarray, 
                            window_length: int = 5) -> np.ndarray:
        """
        Smooth pitch contour using median filtering
        
        Args:
            f0: Fundamental frequency estimates
            window_length: Length of smoothing window (odd number)
            
        Returns:
            Smoothed pitch contour
        """
        from scipy.signal import medfilt
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Handle NaN values
        valid_mask = ~np.isnan(f0)
        smoothed = f0.copy()
        
        if np.any(valid_mask):
            # Only smooth valid values
            valid_f0 = f0[valid_mask]
            smoothed_valid = medfilt(valid_f0, kernel_size=window_length)
            smoothed[valid_mask] = smoothed_valid
        
        logger.info(f"Smoothed pitch contour with window {window_length}")
        return smoothed
    
    def extract_pitch_contour(self, audio: np.ndarray,
                             smooth: bool = True) -> dict:
        """
        Extract complete pitch information from audio
        
        Args:
            audio: Audio signal
            smooth: Apply smoothing if True
            
        Returns:
            Dictionary containing pitch information
        """
        # Detect pitch
        f0, voiced_flag, voiced_probs = self.detect_pitch_pyin(audio)
        
        # Smooth if requested
        if smooth:
            f0 = self.smooth_pitch_contour(f0)
        
        # Convert to MIDI
        midi_notes = self.frequency_to_midi(f0)
        
        # Get time stamps
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        pitch_info = {
            'f0': f0,
            'midi_notes': midi_notes,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'times': times
        }
        
        logger.info("Extracted complete pitch contour")
        return pitch_info
    
    def quantize_pitch(self, midi_notes: np.ndarray) -> np.ndarray:
        """
        Quantize MIDI notes to nearest semitone
        
        Args:
            midi_notes: MIDI note numbers (can contain floats)
            
        Returns:
            Quantized MIDI notes (integers)
        """
        quantized = np.round(midi_notes)
        return quantized
    
    def filter_short_notes(self, midi_notes: np.ndarray, 
                          min_duration_frames: int = 3) -> np.ndarray:
        """
        Filter out very short notes (likely noise)
        
        Args:
            midi_notes: MIDI note sequence
            min_duration_frames: Minimum note duration in frames
            
        Returns:
            Filtered MIDI notes
        """
        filtered = midi_notes.copy()
        
        # Find note boundaries
        note_changes = np.diff(np.concatenate([[np.nan], midi_notes, [np.nan]]))
        note_starts = np.where(note_changes != 0)[0]
        
        # Check duration of each note
        for i in range(len(note_starts) - 1):
            start = note_starts[i]
            end = note_starts[i + 1]
            duration = end - start
            
            # If too short, mark as silence/rest
            if duration < min_duration_frames:
                filtered[start:end] = np.nan
        
        logger.info(f"Filtered short notes (< {min_duration_frames} frames)")
        return filtered
    
    def get_pitch_statistics(self, f0: np.ndarray) -> dict:
        """
        Calculate statistics of pitch contour
        
        Args:
            f0: Fundamental frequency estimates
            
        Returns:
            Dictionary of pitch statistics
        """
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            return {
                'mean_f0': np.nan,
                'std_f0': np.nan,
                'min_f0': np.nan,
                'max_f0': np.nan,
                'range_semitones': np.nan,
                'voiced_ratio': 0.0
            }
        
        mean_f0 = np.mean(valid_f0)
        std_f0 = np.std(valid_f0)
        min_f0 = np.min(valid_f0)
        max_f0 = np.max(valid_f0)
        
        # Calculate range in semitones
        min_midi = librosa.hz_to_midi(min_f0)
        max_midi = librosa.hz_to_midi(max_f0)
        range_semitones = max_midi - min_midi
        
        # Voiced ratio
        voiced_ratio = len(valid_f0) / len(f0)
        
        stats = {
            'mean_f0': mean_f0,
            'std_f0': std_f0,
            'min_f0': min_f0,
            'max_f0': max_f0,
            'range_semitones': range_semitones,
            'voiced_ratio': voiced_ratio
        }
        
        logger.info(f"Pitch statistics: mean={mean_f0:.2f}Hz, range={range_semitones:.1f} semitones")
        return stats
