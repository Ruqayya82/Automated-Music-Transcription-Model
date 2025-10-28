# Automatic Music Transcription

## Project Overview
A machine learning system that transcribes monophonic audio (single instrument, non-percussive) into MIDI and sheet music (MusicXML) format.

## Features
- Audio upload and preprocessing
- Pitch and onset detection using ML
- MIDI file generation
- MusicXML conversion for sheet music
- Web interface for easy interaction
- Visualization and analysis tools

## Target Audience
- Musicians and music producers
- AI/ML researchers in audio processing
- Developers interested in music technology

## Success Metrics
- F1-Score: > 0.80 on model performance
- High sheet music accuracy
- Acceptable processing speed

## Tech Stack
- **Python**: Core language
- **Librosa**: Audio processing and pitch tracking
- **NumPy/SciPy**: Numerical operations and signal processing
- **TensorFlow/PyTorch**: Machine learning models
- **PrettyMIDI**: MIDI generation
- **Music21**: MIDI to MusicXML conversion
- **Flask**: Web framework
- **Jupyter Notebook**: Development and visualization

## Project Structure
```
TranscribeAI/
├── data/                    # Dataset storage
│   ├── raw/                 # Raw audio files
│   ├── processed/           # Preprocessed audio
│   └── ground_truth/        # MIDI ground truth files
├── models/                  # Trained models
│   └── checkpoints/         # Model checkpoints
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/                     # Source code
│   ├── audio_processing/    # Audio processing modules
│   ├── models/              # ML model definitions
│   ├── midi_generation/     # MIDI creation
│   ├── musicxml_conversion/ # MusicXML generation
│   └── web_app/             # Web interface
├── tests/                   # Unit tests
├── outputs/                 # Generated MIDI and MusicXML files
├── figma_designs/           # UI wireframes and mockups
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── config.yaml              # Configuration file
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TranscribeAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
# Transcribe audio file
python src/transcribe.py --input audio.wav --output output.mid

# Generate sheet music
python src/generate_sheet.py --input output.mid --output sheet.xml
```

### Web Interface
```bash
# Start the web server
python src/web_app/app.py

# Navigate to http://localhost:5000
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Model Training

```bash
# Train the transcription model
python src/train.py --config config.yaml
```

## Evaluation

The model is evaluated using:
- F1-Score for pitch and onset detection
- Note accuracy metrics
- Timing precision

## License
MIT License

## Contributors
Your Name

## Acknowledgments
- Librosa documentation
- Music21 library
- TensorFlow/PyTorch communities
