"""
Simple example script for transcribing audio to MIDI

Usage:
    python examples/transcribe_example.py
"""

import sys
sys.path.append('../src')

from models.transcription_model import TranscriptionModel
from midi_generation import MIDICreator
from musicxml_conversion import MusicXMLGenerator
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize transcription model
    logger.info("Initializing transcription model...")
    model = TranscriptionModel(config, device='cpu')
    
    # Example audio file (replace with your own)
    audio_path = '../data/raw/sample.wav'
    
    # Transcribe using traditional methods (no ML model required)
    logger.info(f"Transcribing: {audio_path}")
    notes, metadata = model.transcribe(audio_path, use_model=False)
    
    logger.info(f"Detected {len(notes)} notes")
    
    # Create MIDI file
    midi_creator = MIDICreator(tempo=120, velocity=100)
    midi_path = '../outputs/example_output.mid'
    midi_creator.create_midi(notes, midi_path)
    logger.info(f"MIDI file saved: {midi_path}")
    
    # Create MusicXML file
    xml_generator = MusicXMLGenerator(
        title="Example Transcription",
        composer="TranscribeAI"
    )
    xml_path = '../outputs/example_output.musicxml'
    xml_generator.midi_to_musicxml(midi_path, xml_path)
    logger.info(f"MusicXML file saved: {xml_path}")
    
    logger.info("Transcription complete!")


if __name__ == '__main__':
    main()
