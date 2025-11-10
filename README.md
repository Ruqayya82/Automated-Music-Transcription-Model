# 🎵 TranscribeAI

An AI-powered music transcription system that converts audio recordings of piano and guitar music into MIDI files and MusicXML notation.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** package manager
- **FFmpeg** (required for audio processing)

### System Dependencies

#### Install FFmpeg:

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# OR download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TranscribeAI
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
   *Note: This will install all required packages including Flask, librosa, PyTorch, mido, music21, and more.*

3. **(Optional)** Verify all dependencies are installed:
```bash
python check_dependencies.py
```
   *This script will check if all required packages are properly installed and report any missing dependencies.*

4. Run the application:
```bash
python run.py
```

5. Open your browser to: **http://localhost:5000**


### Troubleshooting Installation

If you encounter "ModuleNotFoundError" after installation:

1. Run the dependency checker to identify missing packages:
   ```bash
   python check_dependencies.py
   ```

2. Install any missing packages individually:
   ```bash
   pip install <package-name>
   ```

3. Ensure FFmpeg is installed (required for audio processing)

4. If issues persist, try upgrading pip:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

## 📁 Project Structure

```
TranscribeAI/
│
├── run.py                          # Main application entry point - RUN THIS FILE
├── config.yaml                     # Configuration settings
├── requirements.txt                # Python dependencies
│
├── data/                           # Audio data directory
│   ├── PianoMusic/                 # Piano audio samples (.mp3)
│   └── GuitarMusic/                # Guitar audio samples (.mp3)
│
├── src/                            # Backend source code
│   ├── audio_processing/           # Audio analysis modules
│   │   ├── audio_loader.py         # Load and preprocess audio
│   │   ├── feature_extractor.py    # Extract audio features
│   │   ├── pitch_detector.py       # Detect pitch
│   │   └── onset_detector.py       # Detect note onsets
│   │
│   ├── models/                     # Transcription models
│   │   ├── transcription_model.py  # Main transcription interface
│   │   ├── pitch_onset_cnn.py      # Optional ML model (not required)
│   │   └── model_trainer.py        # Model training utilities
│   │
│   ├── midi_generation/            # MIDI file generation
│   │   ├── midi_creator.py         # Create MIDI files
│   │   └── note_processor.py       # Process detected notes
│   │
│   ├── musicxml_conversion/        # MusicXML generation
│   │   └── musicxml_generator.py   # Convert MIDI to MusicXML
│   │
│   └── web_app/                    # Frontend (Flask web application)
│       ├── app.py                  # Flask application
│       ├── templates/              # HTML templates
│       │   └── index.html          # Main web interface
│       └── static/                 # Static assets
│           ├── css/
│           │   └── style.css       # Dark mode styling
│           └── js/
│               └── app.js          # Frontend JavaScript
│
├── uploads/                        # Uploaded audio files (auto-created)
├── outputs/                        # Generated MIDI/MusicXML files (auto-created)
│
├── examples/                       # Usage examples
│   └── transcribe_example.py       # Example transcription script
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb
│
├── figma_designs/                  # UI design files
│   ├── wireframe.html
│   └── README.md
│
└── logo/                           # Project branding
    └── TransribeAI logo.png
```

## 🎯 How It Works

TranscribeAI uses **traditional signal processing methods** by default (no ML model required):

1. **Audio Upload** - Upload piano or guitar audio files (.mp3, .wav, .ogg, .flac)
2. **Pitch Detection** - Analyzes the audio to detect musical pitches
3. **Onset Detection** - Identifies when notes start
4. **Note Extraction** - Combines pitch and onset information to extract individual notes
5. **MIDI Generation** - Creates a MIDI file from detected notes
6. **MusicXML Export** - Converts MIDI to MusicXML for use in notation software

## 🎨 Features

- **Dark Mode UI** - Modern, eye-friendly dark theme with proper contrast
- **Drag & Drop Upload** - Easy file uploading interface
- **Real-time Progress** - Visual feedback during transcription
- **Multiple Export Formats** - MIDI and MusicXML output
- **No ML Model Required** - Works out-of-the-box with signal processing
- **Clean Project Structure** - Clearly organized codebase

## 🎼 Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- FLAC (.flac)

## 🔧 Configuration

Edit `config.yaml` to customize:
- Audio processing parameters
- MIDI settings (tempo, velocity)
- Upload/output folders
- File size limits

## 📊 Testing with Sample Data

The `data/` folder contains sample piano and guitar music files you can use to test the system:
- `data/PianoMusic/` - 10 piano audio samples
- `data/GuitarMusic/` - 10 guitar audio samples

## 🛠️ Technical Stack

- **Backend**: Python, Flask
- **Audio Processing**: librosa, numpy, scipy
- **MIDI Generation**: mido
- **MusicXML**: music21
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework** (optional): PyTorch

## 📝 Usage Example

### Using the Web Interface
1. Run `python run.py`
2. Open http://localhost:5000 in your browser
3. Upload an audio file
4. Click "Transcribe"
5. Download MIDI and/or MusicXML files

### Using Python API
```python
from src.models.transcription_model import TranscriptionModel
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = TranscriptionModel(config, device='cpu')

# Transcribe audio
notes, metadata = model.transcribe('data/PianoMusic/piano1.mp3')

print(f"Detected {len(notes)} notes")
```

## 🚧 Development

### Adding New Features
- Audio processing: Add to `src/audio_processing/`
- Models: Add to `src/models/`
- Web interface: Modify `src/web_app/`

### Running Tests
```bash
# In the Future: We Add test commands here
```


## 👥 Contributors

Ruqayya Mustafa

Yuki Li

Patience IZERE

Md Mazharul Islam 

## 🙏 Acknowledgments

Built with librosa, mido, music21, Flask, and PyTorch.
