"""
Onset detection module for identifying note boundaries.
"""

import librosa
import numpy as np
from typing import Tuple, List
import yaml


class OnsetDetector:
    """Detect note onsets (attack times) in audio signals."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize OnsetDetector with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sr = self.config['audio']['sample_rate']
        self.hop_length = self.config['onset_detection']['hop_length']
        self.method = self.config['onset_detection']['method']
        
        # Peak picking parameters
        self.pre_max = self.config['onset_detection']['pre_max']
        self.post_max = self.config['onset_detection']['post_max']
        self.pre_avg = self.config['onset_detection']['pre_avg']
        self.post_avg = self.config['onset_detection']['post_avg']
        self.delta = self.config['onset_detection']['delta']
        self.wait = self.config['onset_detection']['wait']
    
    def compute_onset_strength(self, audio: np.ndarray, 
                               method: str = None) -> np.ndarray:
        """
        Compute onset strength envelope.
        
        Args:
            audio: Input audio signal
            method: Detection method ('spectral_flux', 'energy', 'hfc', etc.)
            
        Returns:
            Onset strength envelope
        """
        if method is None:
            method = self.method
        
        # Compute onset strength using specified method
        if method == 'spectral_flux':
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                aggregate=np.median
            )
        elif method == 'energy':
            # Use RMS energy as onset strength
            onset_env = librosa.feature.rms(
                y=audio,
                hop_length=self.hop_length
            )[0]
        elif method == 'hfc':
            # High Frequency Content
            S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
            freqs = librosa.fft_frequencies(sr=self.sr)
            onset_env = np.sum(S * freqs[:, np.newaxis], axis=0)
        else:
            # Default to spectral flux
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length
            )
        
        return onset_env
    
    def detect_onsets(self, audio: np.ndarray, 
                     backtrack: bool = True) -> np.ndarray:
        """
        Detect onset times in audio.
        
        Args:
            audio: Input audio signal
            backtrack: Whether to backtrack detected onsets to preceding minima
            
        Returns:
            Array of onset times in seconds
        """
        # Compute onset strength
        onset_env = self.compute_onset_strength(audio)
        
        # Detect onset frames
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=backtrack,
            pre_max=self.pre_max,
            post_max=self.post_max,
            pre_avg=self.pre_avg,
            post_avg=self.post_avg,
            delta=self.delta,
            wait=self.wait
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        return onset_times
    
    def detect_onset_frames(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect onset frame indices.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Array of onset frame indices
        """
        onset_env = self.compute_onset_strength(audio)
        
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            pre_max=self.pre_max,
            post_max=self.post_max,
            pre_avg=self.pre_avg,
            post_avg=self.post_avg,
            delta=self.delta,
            wait=self.wait
        )
        
        return onset_frames
    
    def segment_notes(self, audio: np.ndarray, 
                     onset_times: np.ndarray,
                     min_duration: float = 0.1) -> List[Tuple[float, float]]:
        """
        Segment audio into individual notes based on onsets.
        
        Args:
            audio: Input audio signal
            onset_times: Array of onset times
            min_duration: Minimum note duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each note
        """
        segments = []
        
        # Add initial segment if first onset is not at the beginning
        if len(onset_times) > 0 and onset_times[0] > 0.1:
            segments.append((0, onset_times[0]))
        
        # Create segments between consecutive onsets
        for i in range(len(onset_times)):
            start_time = onset_times[i]
            
            if i < len(onset_times) - 1:
                end_time = onset_times[i + 1]
            else:
                # Last segment extends to end of audio
                end_time = len(audio) / self.sr
            
            # Only add segment if it meets minimum duration
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
        
        return segments
    
    def refine_onsets_with_pitch(self, onset_times: np.ndarray,
                                 pitch_times: np.ndarray,
                                 midi_notes: np.ndarray) -> np.ndarray:
        """
        Refine onset times based on pitch changes.
        
        Args:
            onset_times: Initial onset times
            pitch_times: Times corresponding to pitch estimates
            midi_notes: MIDI note values at each pitch time
            
        Returns:
            Refined onset times
        """
        # Detect pitch changes
        pitch_changes = np.where(np.diff(midi_notes) != 0)[0]
        pitch_change_times = pitch_times[pitch_changes]
        
        # Combine and sort onset times with pitch change times
        combined_onsets = np.unique(np.concatenate([onset_times, pitch_change_times]))
        
        return combined_onsets
    
    def get_note_durations(self, onset_times: np.ndarray, 
                          audio_duration: float) -> np.ndarray:
        """
        Calculate note durations from onset times.
        
        Args:
            onset_times: Array of onset times
            audio_duration: Total duration of audio in seconds
            
        Returns:
            Array of note durations in seconds
        """
        durations = np.zeros(len(onset_times))
        
        for i in range(len(onset_times)):
            if i < len(onset_times) - 1:
                durations[i] = onset_times[i + 1] - onset_times[i]
            else:
                durations[i] = audio_duration - onset_times[i]
        
        return durations
    
    def quantize_onsets_to_grid(self, onset_times: np.ndarray,
                                tempo: float = 120,
                                subdivision: int = 16) -> np.ndarray:
        """
        Quantize onset times to a rhythmic grid.
        
        Args:
            onset_times: Array of onset times
            tempo: Tempo in BPM
            subdivision: Subdivision of beat (16 = sixteenth notes)
            
        Returns:
            Quantized onset times
        """
        # Calculate grid spacing
        beat_duration = 60.0 / tempo
        grid_spacing = beat_duration / (subdivision / 4)
        
        # Quantize each onset to nearest grid point
        quantized_onsets = np.round(onset_times / grid_spacing) * grid_spacing
        
        return quantized_onsets
