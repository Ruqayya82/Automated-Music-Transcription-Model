"""
Main transcription pipeline that integrates all components.
"""

import numpy as np
from typing import Tuple, Optional
import os
import yaml

from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor
from models.pitch_detector import PitchDetector
from models.onset_detector import OnsetDetector
from postprocessing.midi_generator import MIDIGenerator
from postprocessing.musicxml_generator import MusicXMLGenerator


class TranscriptionPipeline:
    """Complete pipeline for automatic music transcription."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the transcription pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.audio_loader = AudioLoader(config_path)
        self.feature_extractor = FeatureExtractor(config_path)
        self.pitch_detector = PitchDetector(config_path)
        self.onset_detector = OnsetDetector(config_path)
        self.midi_generator = MIDIGenerator(config_path)
        self.musicxml_generator = MusicXMLGenerator(config_path)
        
        # Create output directory if it doesn't exist
        self.output_dir = self.config['paths']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def transcribe(self,
                   audio_path: str,
                   output_name: str = None,
                   save_midi: bool = True,
                   save_musicxml: bool = True) -> Tuple[str, str]:
        """
        Complete transcription pipeline from audio to sheet music.
        
        Args:
            audio_path: Path to input audio file
            output_name: Base name for output files (without extension)
            save_midi: Whether to save MIDI file
            save_musicxml: Whether to save MusicXML file
            
        Returns:
            Tuple of (midi_path, musicxml_path)
        """
        print("\n" + "="*60)
        print("TranscribeAI - Automatic Music Transcription")
        print("="*60 + "\n")
        
        # Step 1: Load and preprocess audio
        print("Step 1/5: Loading and preprocessing audio...")
        audio, sr = self.audio_loader.preprocess(audio_path)
        print(f"✓ Audio loaded successfully\n")
        
        # Step 2: Detect pitch
        print("Step 2/5: Detecting pitch...")
        frequencies, voiced_flags, midi_notes = self.pitch_detector.detect_and_quantize(audio)
        times = self.pitch_detector.get_pitch_times(audio)
        print(f"✓ Detected {np.sum(voiced_flags)} voiced frames\n")
        
        # Step 3: Detect onsets
        print("Step 3/5: Detecting note onsets...")
        onset_times = self.onset_detector.detect_onsets(audio)
        print(f"✓ Detected {len(onset_times)} note onsets\n")
        
        # Step 4: Generate MIDI
        print("Step 4/5: Generating MIDI file...")
        midi = self.midi_generator.create_midi_from_pitch_contour(
            midi_notes, times, voiced_flags
        )
        
        # Determine output paths
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        midi_path = None
        if save_midi:
            midi_path = os.path.join(self.output_dir, f"{output_name}.mid")
            self.midi_generator.save_midi(midi, midi_path)
            print(f"✓ MIDI file created\n")
        
        # Step 5: Generate MusicXML
        print("Step 5/5: Generating sheet music...")
        musicxml_path = None
        if save_musicxml and midi_path:
            musicxml_path = os.path.join(self.output_dir, f"{output_name}.musicxml")
            score = self.musicxml_generator.midi_to_musicxml(
                midi_path,
                title=output_name,
                composer="TranscribeAI"
            )
            self.musicxml_generator.save_musicxml(score, musicxml_path)
            print(f"✓ MusicXML file created\n")
        
        print("="*60)
        print("Transcription completed successfully!")
        print("="*60 + "\n")
        
        return midi_path, musicxml_path
    
    def transcribe_with_onsets(self,
                               audio_path: str,
                               output_name: str = None,
                               quantize: bool = True) -> Tuple[str, str]:
        """
        Transcription using onset detection for note segmentation.
        
        Args:
            audio_path: Path to input audio file
            output_name: Base name for output files
            quantize: Whether to quantize timings to grid
            
        Returns:
            Tuple of (midi_path, musicxml_path)
        """
        print("\n" + "="*60)
        print("TranscribeAI - Onset-Based Transcription")
        print("="*60 + "\n")
        
        # Load audio
        print("Loading audio...")
        audio, sr = self.audio_loader.preprocess(audio_path)
        
        # Detect pitch
        print("Detecting pitch...")
        frequencies, voiced_flags, midi_notes = self.pitch_detector.detect_and_quantize(audio)
        times = self.pitch_detector.get_pitch_times(audio)
        
        # Detect onsets
        print("Detecting onsets...")
        onset_times = self.onset_detector.detect_onsets(audio)
        
        # Refine onsets with pitch changes
        onset_times = self.onset_detector.refine_onsets_with_pitch(
            onset_times, times, midi_notes
        )
        
        # Get note durations
        audio_duration = len(audio) / sr
        durations = self.onset_detector.get_note_durations(onset_times, audio_duration)
        
        # Get pitch at each onset
        onset_pitches = []
        for onset_time in onset_times:
            # Find closest pitch estimate
            time_idx = np.argmin(np.abs(times - onset_time))
            onset_pitches.append(midi_notes[time_idx])
        
        onset_pitches = np.array(onset_pitches)
        
        # Quantize if requested
        if quantize:
            tempo = self.midi_generator.estimate_tempo(onset_times)
            onset_times, durations = self.midi_generator.quantize_timing(
                onset_times, durations, tempo
            )
        
        # Generate MIDI
        print("Generating MIDI...")
        midi = self.midi_generator.create_midi_from_notes(
            onset_pitches, onset_times, durations
        )
        
        # Save files
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(audio_path))[0] + "_onset"
        
        midi_path = os.path.join(self.output_dir, f"{output_name}.mid")
        self.midi_generator.save_midi(midi, midi_path)
        
        musicxml_path = os.path.join(self.output_dir, f"{output_name}.musicxml")
        score = self.musicxml_generator.midi_to_musicxml(
            midi_path, title=output_name
        )
        self.musicxml_generator.save_musicxml(score, musicxml_path)
        
        print("\nTranscription completed!\n")
        
        return midi_path, musicxml_path
    
    def analyze_audio(self, audio_path: str) -> dict:
        """
        Analyze audio and return diagnostic information.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        # Load audio
        audio, sr = self.audio_loader.preprocess(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio)
        
        # Detect pitch
        frequencies, voiced_flags, midi_notes = self.pitch_detector.detect_and_quantize(audio)
        
        # Detect onsets
        onset_times = self.onset_detector.detect_onsets(audio)
        
        # Compile analysis
        analysis = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'num_samples': len(audio),
            'num_voiced_frames': int(np.sum(voiced_flags)),
            'pitch_range': {
                'min_hz': float(np.min(frequencies[voiced_flags])) if np.any(voiced_flags) else 0,
                'max_hz': float(np.max(frequencies[voiced_flags])) if np.any(voiced_flags) else 0,
                'min_midi': float(np.min(midi_notes[midi_notes > 0])) if np.any(midi_notes > 0) else 0,
                'max_midi': float(np.max(midi_notes[midi_notes > 0])) if np.any(midi_notes > 0) else 0,
            },
            'num_onsets': len(onset_times),
            'estimated_tempo': float(self.midi_generator.estimate_tempo(onset_times)) if len(onset_times) > 1 else 0,
            'features': {
                'spectrogram_shape': features['spectrogram'].shape,
                'mel_spectrogram_shape': features['mel_spectrogram'].shape,
            }
        }
        
        return analysis
    
    def get_visualization_data(self, audio_path: str) -> dict:
        """
        Get data for visualization purposes.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with visualization data
        """
        # Load audio
        audio, sr = self.audio_loader.preprocess(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio)
        
        # Detect pitch
        frequencies, voiced_flags, midi_notes = self.pitch_detector.detect_and_quantize(audio)
        times = self.pitch_detector.get_pitch_times(audio)
        
        # Detect onsets
        onset_times = self.onset_detector.detect_onsets(audio)
        onset_strength = self.onset_detector.compute_onset_strength(audio)
        
        viz_data = {
            'audio': audio,
            'sample_rate': sr,
            'times': times,
            'frequencies': frequencies,
            'voiced_flags': voiced_flags,
            'midi_notes': midi_notes,
            'onset_times': onset_times,
            'onset_strength': onset_strength,
            'spectrogram': features['spectrogram'],
            'mel_spectrogram': features['mel_spectrogram'],
            'chromagram': features['chromagram'],
        }
        
        return viz_data
