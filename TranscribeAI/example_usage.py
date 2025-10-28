"""
Example script demonstrating how to use TranscribeAI.
"""

from models.transcription_model import TranscriptionPipeline
import os


def main():
    """Run a simple transcription example."""
    
    # Initialize the pipeline
    print("Initializing TranscribeAI pipeline...")
    pipeline = TranscriptionPipeline(config_path='config.yaml')
    
    # Example 1: Basic transcription
    print("\n" + "="*60)
    print("Example 1: Basic Transcription")
    print("="*60)
    
    # Note: You'll need to provide your own audio file
    audio_file = 'data/sample_audio/example.wav'
    
    if os.path.exists(audio_file):
        midi_path, musicxml_path = pipeline.transcribe(
            audio_file,
            output_name='example_transcription',
            save_midi=True,
            save_musicxml=True
        )
        
        print(f"\nOutput files:")
        print(f"  MIDI: {midi_path}")
        print(f"  MusicXML: {musicxml_path}")
    else:
        print(f"Audio file not found: {audio_file}")
        print("Please add a monophonic audio file to test the transcription.")
    
    # Example 2: Onset-based transcription with quantization
    print("\n" + "="*60)
    print("Example 2: Onset-Based Transcription")
    print("="*60)
    
    if os.path.exists(audio_file):
        midi_path, musicxml_path = pipeline.transcribe_with_onsets(
            audio_file,
            output_name='example_onset_transcription',
            quantize=True
        )
        
        print(f"\nOutput files:")
        print(f"  MIDI: {midi_path}")
        print(f"  MusicXML: {musicxml_path}")
    
    # Example 3: Audio analysis
    print("\n" + "="*60)
    print("Example 3: Audio Analysis")
    print("="*60)
    
    if os.path.exists(audio_file):
        analysis = pipeline.analyze_audio(audio_file)
        
        print(f"\nAudio Analysis:")
        print(f"  Duration: {analysis['duration']:.2f} seconds")
        print(f"  Sample Rate: {analysis['sample_rate']} Hz")
        print(f"  Voiced Frames: {analysis['num_voiced_frames']}")
        print(f"  Detected Notes: {analysis['num_onsets']}")
        print(f"  Estimated Tempo: {analysis['estimated_tempo']:.0f} BPM")
        
        if analysis['pitch_range']['min_hz'] > 0:
            print(f"\n  Pitch Range:")
            print(f"    Frequency: {analysis['pitch_range']['min_hz']:.1f} - "
                  f"{analysis['pitch_range']['max_hz']:.1f} Hz")
            print(f"    MIDI Notes: {analysis['pitch_range']['min_midi']:.0f} - "
                  f"{analysis['pitch_range']['max_midi']:.0f}")


if __name__ == "__main__":
    main()
