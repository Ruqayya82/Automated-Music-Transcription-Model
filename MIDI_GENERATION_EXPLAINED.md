# MIDI Generation Process - Technical Documentation

## Overview

TranscribeAI generates **real MIDI files from audio transcription**, not audio-to-MIDI conversion. This document explains how the process works.

## The Process

### 1. Audio Analysis (Transcription)
When you upload an audio file, the system:

- **Loads and preprocesses the audio** using librosa
- **Detects pitch** using advanced algorithms (PYIN, CREPE, or YIN)
- **Detects note onsets** (when notes start) using onset detection algorithms
- **Extracts musical notes** with precise timing, pitch, and duration information

This is **NOT** a simple audio-to-MIDI conversion. The system analyzes the actual musical content and extracts individual notes.

### 2. Note Extraction
The transcription engine identifies:
- **Start time** of each note (in seconds)
- **Duration** of each note (in seconds)
- **Pitch** (as MIDI note number, e.g., 60 = Middle C)
- **Velocity** (note intensity, 0-127)

Example note data:
```python
{
    'start_time': 0.5,      # Note starts at 0.5 seconds
    'duration': 0.25,       # Note lasts 0.25 seconds
    'pitch': 60,            # Middle C
    'velocity': 80          # Medium-loud
}
```

### 3. MIDI File Creation with PrettyMIDI

The system uses **PrettyMIDI** library to create proper MIDI files from the detected notes:

```python
# Create PrettyMIDI object
midi = pretty_midi.PrettyMIDI(initial_tempo=120)

# Create instrument
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

# Add each detected note
for note_data in notes:
    note = pretty_midi.Note(
        velocity=note_data['velocity'],
        pitch=note_data['pitch'],
        start=note_data['start_time'],
        end=note_data['start_time'] + note_data['duration']
    )
    instrument.notes.append(note)

# Add instrument to MIDI and save
midi.instruments.append(instrument)
midi.write('output.mid')
```

## Key Differences from Audio Conversion

### ❌ Audio-to-MIDI Conversion (What we DON'T do)
- Simply wraps audio data in MIDI format
- No musical analysis
- Cannot be edited as musical notes
- Sounds like the original audio when played

### ✅ True Transcription (What we DO)
- Analyzes audio to detect individual notes
- Extracts pitch, timing, and duration
- Creates proper MIDI with note events
- Can be edited in any MIDI editor
- Sounds like a MIDI instrument when played

## How It Works in the App

1. **Upload Audio**: User uploads WAV, MP3, FLAC, OGG, or M4A
2. **Analyze Audio**: System detects pitches and note onsets
3. **Extract Notes**: Converts analysis into note list with timing
4. **Generate MIDI**: PrettyMIDI creates proper MIDI file from notes
5. **Generate MusicXML**: Music21 converts MIDI to sheet music format

## Methods

### Traditional Signal Processing (Default)
- Uses librosa's pitch detection (PYIN algorithm)
- Uses onset detection for note timing
- Fast and works well for monophonic audio
- No ML model required

### ML Model (Optional)
- Uses CNN-LSTM neural network
- More accurate for complex audio
- Requires trained model weights
- Slower but better results

## Code Architecture

### Key Components

**`TranscriptionModel`** (`src/models/transcription_model.py`)
- Handles audio analysis
- Detects pitches and onsets
- Extracts note information

**`MIDICreator`** (`src/midi_generation/midi_creator.py`)
- Uses PrettyMIDI library
- Creates MIDI files from note lists
- Supports tempo, time signatures, key signatures

**`Flask App`** (`src/web_app/app.py`)
- Coordinates the transcription pipeline
- Validates MIDI file creation
- Provides download endpoints

## Example Workflow

```
Audio File (piano.mp3)
    ↓
[Pitch Detection]
    ↓
Detected Pitches: [C4, D4, E4, F4, G4...]
    ↓
[Onset Detection]
    ↓
Note Timings: [0.0s, 0.5s, 1.0s, 1.5s, 2.0s...]
    ↓
[Note Extraction]
    ↓
Notes List: [
    {pitch: 60, start: 0.0, duration: 0.5},
    {pitch: 62, start: 0.5, duration: 0.5},
    {pitch: 64, start: 1.0, duration: 0.5},
    ...
]
    ↓
[PrettyMIDI Generation]
    ↓
MIDI File (piano.mid)
```

## Technical Details

### Libraries Used
- **librosa**: Audio processing and feature extraction
- **PrettyMIDI**: MIDI file creation and manipulation
- **NumPy**: Numerical computations
- **PyTorch**: ML model (optional)
- **Music21**: MusicXML generation

### MIDI Standards Compliance
- Standard MIDI File (SMF) format
- Type 0 or Type 1 MIDI files
- Tempo and time signature support
- Key signature support
- Standard MIDI note numbers (0-127)

### Supported Features
- ✅ Note pitch (MIDI note numbers)
- ✅ Note timing (start time, duration)
- ✅ Note velocity (loudness)
- ✅ Tempo changes
- ✅ Time signatures
- ✅ Key signatures
- ✅ Multiple instruments (planned)
- ❌ Polyphony (currently monophonic only)
- ❌ Dynamics/articulations (planned)

## Validation

The app validates MIDI generation:
```python
# Create MIDI file from transcribed notes
midi_obj = app.midi_creator.create_midi(notes, midi_path)

# Validate MIDI file was created properly
if not os.path.exists(midi_path):
    raise Exception("MIDI file generation failed")

# Get MIDI file info for metadata
midi_info = app.midi_creator.get_midi_info(midi_path)
logger.info(f"Generated MIDI with {midi_info['total_notes']} notes")
```

## Output Quality

The quality of the MIDI output depends on:
1. **Input audio quality**: Clear, monophonic audio works best
2. **Instrument type**: Piano and string instruments work well
3. **Background noise**: Clean recordings produce better results
4. **Transcription method**: ML model is more accurate but slower

## Future Enhancements

Planned improvements:
- [ ] Polyphonic transcription (multiple simultaneous notes)
- [ ] Better rhythm quantization
- [ ] Chord detection
- [ ] Multi-instrument support
- [ ] Real-time transcription
- [ ] Fine-tuned ML models for specific instruments

## Conclusion

TranscribeAI creates **genuine MIDI files** through audio analysis and note extraction, powered by PrettyMIDI. The MIDI files can be:
- Opened in any DAW (Digital Audio Workstation)
- Edited in MIDI editors like MuseScore, Finale, Sibelius
- Played with any MIDI instrument
- Converted to sheet music
- Transposed, quantized, and modified freely

This is true music transcription, not audio conversion!
