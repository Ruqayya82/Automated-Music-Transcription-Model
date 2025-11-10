# TranscribeAI - Project Structure Guide

This document explains the organization of the TranscribeAI project for easy navigation and understanding.

## 📂 Project Organization

### Root Level Files
- **run.py** - 🚀 MAIN APPLICATION ENTRY POINT - Run this to start the server
- **config.yaml** - Configuration settings for audio processing, MIDI, and web app
- **requirements.txt** - Python package dependencies
- **setup.py** - Package installation script
- **README.md** - Project documentation and quick start guide
- **QUICKSTART.md** - Quick start instructions

### Data Directory (`data/`)
📁 **Audio sample files for testing**
- `PianoMusic/` - 10 piano audio samples (piano1.mp3 through piano10.mp3)
- `GuitarMusic/` - 10 guitar audio samples (guitar1.mp3 through guitar10.mp3)

### Source Code (`src/`)
🔧 **All backend Python code**

#### Audio Processing (`src/audio_processing/`)
Signal processing modules for audio analysis:
- `audio_loader.py` - Load and preprocess audio files
- `feature_extractor.py` - Extract audio features (mel spectrograms, etc.)
- `pitch_detector.py` - Detect musical pitches
- `onset_detector.py` - Detect note onset times

#### Models (`src/models/`)
Transcription model code:
- `transcription_model.py` - Main transcription interface (uses traditional signal processing by default)
- `pitch_onset_cnn.py` - Optional ML model architecture
- `model_trainer.py` - Model training utilities

#### MIDI Generation (`src/midi_generation/`)
MIDI file creation:
- `midi_creator.py` - Create MIDI files from detected notes
- `note_processor.py` - Process and quantize musical notes

#### MusicXML Conversion (`src/musicxml_conversion/`)
Sheet music generation:
- `musicxml_generator.py` - Convert MIDI to MusicXML format

#### Web Application (`src/web_app/`)
🌐 **Frontend - Flask web interface**
- `app.py` - Flask application and API endpoints
- `templates/index.html` - Main web page (HTML)
- `static/css/style.css` - Dark mode styling (CSS)
- `static/js/app.js` - Frontend JavaScript
- `static/logo.png` - Application logo

### Working Directories
🗂️ **Auto-created at runtime**
- `uploads/` - Temporary storage for uploaded audio files
- `outputs/` - Generated MIDI and MusicXML files

### Additional Resources
- `examples/` - Example Python scripts showing how to use the API
- `notebooks/` - Jupyter notebooks for data exploration
- `figma_designs/` - UI/UX design files and wireframes
- `logo/` - Project branding assets

## 🎯 Quick Navigation Guide

### To modify the UI:
- **HTML**: `src/web_app/templates/index.html`
- **CSS (Dark Mode)**: `src/web_app/static/css/style.css`
- **JavaScript**: `src/web_app/static/js/app.js`

### To modify backend logic:
- **Main API**: `src/web_app/app.py`
- **Transcription**: `src/models/transcription_model.py`
- **Audio processing**: `src/audio_processing/*.py`

### To add new features:
- **Audio algorithms**: Add to `src/audio_processing/`
- **ML models**: Add to `src/models/`
- **Web endpoints**: Add to `src/web_app/app.py`

### To change settings:
- **All configurations**: Edit `config.yaml`

## 🏃 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run.py
   ```

3. Open browser to: http://localhost:5000

## 📝 File Categories

### Frontend (Web UI)
```
src/web_app/
├── templates/     # HTML
├── static/css/    # Stylesheets (Dark Mode)
└── static/js/     # JavaScript
```

### Backend (Python Logic)
```
src/
├── audio_processing/    # Audio analysis
├── models/              # Transcription models
├── midi_generation/     # MIDI creation
└── musicxml_conversion/ # MusicXML export
```

### Data
```
data/
├── PianoMusic/    # Piano samples
└── GuitarMusic/   # Guitar samples
```

### Configuration & Documentation
```
.
├── run.py           # Entry point
├── config.yaml      # Settings
├── README.md        # Main docs
└── requirements.txt # Dependencies
```

## 🎨 Dark Mode Theme

The UI now features a modern dark theme with:
- **Background**: Deep navy (#0f172a)
- **Cards**: Dark slate (#1e293b)
- **Text**: Light gray (#f1f5f9)
- **Accent**: Indigo (#818cf8)
- **Success**: Emerald (#34d399)
- **Error**: Red (#f87171)

All colors have been carefully selected for proper contrast and readability.

## ⚙️ Model Configuration

The transcription model now works in two modes:
1. **Traditional Signal Processing** (default) - No ML model required, works immediately
2. **ML Model** (optional) - Requires trained weights, more accurate

This makes the app functional right away without needing pretrained model files!
