# TranscribeAI - Automatic Music Transcription

## Project Overview
TranscribeAI is a machine learning model designed to transcribe monophonic audio (single instrument, non-percussive) into MIDI and MusicXML formats, enabling musicians to view and print sheet music from audio recordings.

## Audience
- Musicians seeking to transcribe melodies
- Music producers analyzing compositions
- AI/ML researchers exploring audio signal processing
- Developers building music technology applications

## Key Features
- **Audio Upload**: Support for WAV, MP3, and other common audio formats
- **Pitch Detection**: Advanced pitch tracking using spectral analysis
- **Onset Detection**: Identify note start times accurately
- **MIDI Generation**: Convert detected pitches to MIDI format
- **MusicXML Generation**: Create printable sheet music
- **Web Interface**: User-friendly Streamlit-based interface
- **Visualization**: Display spectrograms and detected notes

## Success Metrics
- **F1-Score**: Target > 0.80 on pitch detection
- **Sheet Music Accuracy**: High fidelity transcription
- **Performance**: Real-time or near real-time processing

## Project Structure
```
TranscribeAI/
├── README.md
├── requirements.txt
├── setup.py
├── config.yaml
├── data/
│   ├── sample_audio/
│   └── datasets/
├── models/
│   ├── __init__.py
│   ├── pitch_detector.py
│   ├── onset_detector.py
│   └── transcription_model.py
├── preprocessing/
│   ├── __init__.py
│   ├── audio_loader.py
│   └── feature_extractor.py
├── postprocessing/
│   ├── __init__.py
│   ├── midi_generator.py
│   └── musicxml_generator.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── web_interface/
│   └── streamlit_app.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── tests/
    └── test_pipeline.py
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup
```bash
# Clone or navigate to the project directory
cd TranscribeAI

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line
```python
from models.transcription_model import TranscriptionPipeline

# Initialize the pipeline
pipeline = TranscriptionPipeline()

# Transcribe audio file
midi_file, musicxml_file = pipeline.transcribe('path/to/audio.wav')
```

### Web Interface
```bash
# Launch the Streamlit app
streamlit run web_interface/streamlit_app.py
```

## Model Architecture

The transcription pipeline consists of:
1. **Audio Preprocessing**: Load and normalize audio, compute spectrograms
2. **Pitch Detection**: Extract fundamental frequency using autocorrelation and spectral methods
3. **Onset Detection**: Identify note boundaries using spectral flux
4. **Note Quantization**: Convert continuous pitch to discrete MIDI notes
5. **MIDI Generation**: Create MIDI file from detected notes
6. **MusicXML Conversion**: Generate printable sheet music

## Technologies Used
- **Python**: Core programming language
- **Librosa**: Audio analysis and feature extraction
- **NumPy/SciPy**: Numerical operations and signal processing
- **TensorFlow/PyTorch**: Deep learning models
- **PrettyMIDI**: MIDI file generation
- **Music21**: MusicXML conversion
- **Streamlit**: Web interface
- **Jupyter**: Interactive development

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## License
MIT License

## Contact
For questions or feedback, please open an issue in the repository.
