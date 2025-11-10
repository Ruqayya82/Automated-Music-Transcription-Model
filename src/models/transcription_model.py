"""
Transcription Model Interface

High-level interface for music transcription
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

from src.audio_processing import AudioLoader, FeatureExtractor, PitchDetector, OnsetDetector
from src.models.pitch_onset_cnn import create_model

logger = logging.getLogger(__name__)


class TranscriptionModel:
    """
    High-level transcription model interface
    
    Combines audio processing and ML model for end-to-end transcription
    """
    
    def __init__(self, config: dict, device: str = 'cpu'):
        """
        Initialize TranscriptionModel
        
        Args:
            config: Configuration dictionary
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = torch.device(device)
        
        # Initialize audio processing components
        audio_config = config['audio']
        self.audio_loader = AudioLoader(
            sample_rate=audio_config['sample_rate']
        )
        
        self.feature_extractor = FeatureExtractor(
            sample_rate=audio_config['sample_rate'],
            n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length'],
            n_mels=audio_config['n_mels']
        )
        
        pitch_config = config['pitch']
        self.pitch_detector = PitchDetector(
            sample_rate=audio_config['sample_rate'],
            hop_length=pitch_config['hop_length'],
            fmin=pitch_config['fmin'],
            fmax=pitch_config['fmax']
        )
        
        onset_config = config['onset']
        self.onset_detector = OnsetDetector(
            sample_rate=audio_config['sample_rate'],
            hop_length=audio_config['hop_length']
        )
        
        # Initialize ML model (optional - only if weights are loaded)
        self.model = None
        
        logger.info(f"Initialized TranscriptionModel on {device} (using traditional signal processing)")
    
    def load_weights(self, checkpoint_path: str):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from {checkpoint_path}")
    
    def save_weights(self, checkpoint_path: str, metadata: Optional[dict] = None):
        """
        Save model weights to checkpoint
        
        Args:
            checkpoint_path: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model weights to {checkpoint_path}")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio
        
        Args:
            audio: Audio signal
            
        Returns:
            Feature array
        """
        audio_config = self.config['audio']
        features = self.feature_extractor.extract_mel_spectrogram(
            audio,
            fmin=audio_config['fmin'],
            fmax=audio_config['fmax']
        )
        
        # Normalize features
        features = self.feature_extractor.normalize_features(features, axis=1)
        
        return features
    
    def transcribe_audio(self, audio_path: str, 
                        use_model: bool = False) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            use_model: Use ML model if True, use traditional methods otherwise (default: False)
            
        Returns:
            Dictionary containing transcription results
        """
        # Load and preprocess audio
        audio, sr = self.audio_loader.preprocess(audio_path)
        
        if use_model and self.model is not None:
            return self._transcribe_with_model(audio)
        else:
            return self._transcribe_traditional(audio)
    
    def _transcribe_with_model(self, audio: np.ndarray) -> Dict:
        """
        Transcribe using ML model
        
        Args:
            audio: Audio signal
            
        Returns:
            Transcription results
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Prepare input for model
        features_tensor = torch.FloatTensor(features.T).unsqueeze(0).to(self.device)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            pitch_out, onset_out = self.model(features_tensor)
        
        # Convert to numpy
        pitch_probs = pitch_out.cpu().numpy()[0]
        onset_probs = onset_out.cpu().numpy()[0]
        
        # Get timestamps
        times = self.feature_extractor.frames_to_time(np.arange(len(pitch_probs)))
        
        result = {
            'pitch_probabilities': pitch_probs,
            'onset_probabilities': onset_probs,
            'times': times,
            'sample_rate': self.config['audio']['sample_rate'],
            'method': 'model'
        }
        
        logger.info("Transcription complete using ML model")
        return result
    
    def _transcribe_traditional(self, audio: np.ndarray) -> Dict:
        """
        Transcribe using traditional signal processing methods
        
        Args:
            audio: Audio signal
            
        Returns:
            Transcription results
        """
        # Pitch detection
        pitch_info = self.pitch_detector.extract_pitch_contour(audio, smooth=True)
        
        # Onset detection
        onsets = self.onset_detector.detect_onsets(audio, units='time')
        
        result = {
            'pitch_info': pitch_info,
            'onsets': onsets,
            'method': 'traditional'
        }
        
        logger.info("Transcription complete using traditional methods")
        return result
    
    def predict_notes(self, transcription_result: Dict,
                     pitch_threshold: float = 0.5,
                     onset_threshold: float = 0.5) -> list:
        """
        Convert transcription result to note list
        
        Args:
            transcription_result: Output from transcribe_audio
            pitch_threshold: Threshold for pitch detection
            onset_threshold: Threshold for onset detection
            
        Returns:
            List of notes with (start_time, duration, pitch) tuples
        """
        if transcription_result['method'] == 'model':
            return self._extract_notes_from_model_output(
                transcription_result, pitch_threshold, onset_threshold
            )
        else:
            return self._extract_notes_from_traditional(transcription_result)
    
    def _extract_notes_from_model_output(self, result: Dict,
                                        pitch_threshold: float,
                                        onset_threshold: float) -> list:
        """Extract notes from model predictions"""
        pitch_probs = result['pitch_probabilities']
        onset_probs = result['onset_probabilities']
        times = result['times']
        
        notes = []
        current_note = None
        
        for i, (time, onset_prob) in enumerate(zip(times, onset_probs)):
            # Detect onset
            if onset_prob > onset_threshold:
                # Save previous note if exists
                if current_note is not None:
                    current_note['duration'] = time - current_note['start_time']
                    notes.append(current_note)
                
                # Get pitch at this onset
                pitch_frame = pitch_probs[i]
                active_pitches = np.where(pitch_frame > pitch_threshold)[0]
                
                if len(active_pitches) > 0:
                    # Take strongest pitch
                    strongest_pitch_idx = active_pitches[np.argmax(pitch_frame[active_pitches])]
                    midi_note = strongest_pitch_idx + 21  # A0 = 21
                    
                    current_note = {
                        'start_time': time,
                        'pitch': int(midi_note),
                        'velocity': 80
                    }
        
        # Add last note
        if current_note is not None:
            current_note['duration'] = times[-1] - current_note['start_time']
            notes.append(current_note)
        
        return notes
    
    def _extract_notes_from_traditional(self, result: Dict) -> list:
        """Extract notes from traditional method output"""
        pitch_info = result['pitch_info']
        onsets = result['onsets']
        
        midi_notes = pitch_info['midi_notes']
        times = pitch_info['times']
        
        notes = []
        
        for i, onset_time in enumerate(onsets):
            # Find closest time index
            onset_idx = np.argmin(np.abs(times - onset_time))
            
            # Get pitch at onset
            midi_note = midi_notes[onset_idx]
            
            if not np.isnan(midi_note):
                # Determine duration (until next onset or end)
                if i < len(onsets) - 1:
                    duration = onsets[i + 1] - onset_time
                else:
                    duration = times[-1] - onset_time
                
                notes.append({
                    'start_time': onset_time,
                    'duration': duration,
                    'pitch': int(np.round(midi_note)),
                    'velocity': 80
                })
        
        return notes
    
    def transcribe(self, audio_path: str, 
                  use_model: bool = False) -> Tuple[list, Dict]:
        """
        Full transcription pipeline
        
        Args:
            audio_path: Path to audio file
            use_model: Use ML model if True (default: False - uses traditional signal processing)
            
        Returns:
            Tuple of (notes_list, metadata)
        """
        import time
        start_time = time.time()
        
        # Transcribe
        result = self.transcribe_audio(audio_path, use_model=use_model)
        
        # Extract notes
        notes = self.predict_notes(result)
        
        processing_time = time.time() - start_time
        
        # Metadata
        metadata = {
            'source_file': audio_path,
            'num_notes': len(notes),
            'method': result['method'],
            'processing_time': round(processing_time, 2)
        }
        
        logger.info(f"Transcribed {len(notes)} notes from {audio_path} in {processing_time:.2f}s using {result['method']} method")
        return notes, metadata
