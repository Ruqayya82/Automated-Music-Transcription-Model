# TranscribeAI - Quick Start Guide

Welcome to TranscribeAI! This guide will help you get started quickly with the music transcription system.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
# Start the Flask web server
python src/web_app/app.py
```

Then open your browser to `http://localhost:5000`

## 📝 Usage Examples

### Web Interface (Recommended for Beginners)

1. Open `http://localhost:5000` in your browser
2. Upload an audio file (WAV, MP3, FLAC, OGG, M4A)
3. Click "Transcribe Audio"
4. Download the generated MIDI and MusicXML files

### Command Line

```bash
# Using the example script
cd examples
python transcribe_example.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_data_exploration.ipynb
```

## 📁 Project Structure

```
TranscribeAI/
├── src/                    # Source code
│   ├── audio_processing/   # Audio analysis
│   ├── models/             # ML models
│   ├── midi_generation/    # MIDI creation
│   ├── musicxml_conversion/# Sheet music
│   └── web_app/            # Web interface
├── notebooks/              # Jupyter notebooks
├── examples/               # Example scripts
├── data/                   # Audio data
├── outputs/                # Generated files
├── figma_designs/          # UI wireframes
└── logo/                   # Brand assets
```

## 🎵 Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

**Best Results**: Monophonic audio (single instrument, no percussion)

## 🔧 Configuration

Edit `config.yaml` to customize:
- Audio processing settings
- Model parameters
- MIDI generation options
- MusicXML formatting

## 📊 Features

### Audio Processing
- ✅ Pitch detection (pYIN algorithm)
- ✅ Onset detection
- ✅ Feature extraction (mel spectrogram, MFCC, etc.)

### Transcription Methods
- **Traditional**: Signal processing (fast, no training required)
- **ML Model**: CNN-LSTM network (more accurate, requires training)

### Output Formats
- **MIDI**: Standard MIDI file format
- **MusicXML**: Sheet music format (open in MuseScore, Finale, etc.)

## 🎯 Workflow

1. **Upload Audio** → Choose your audio file
2. **Select Method** → ML Model or Traditional
3. **Transcribe** → Process the audio
4. **Download** → Get MIDI and MusicXML files
5. **View** → Open in your favorite music software

## 💡 Tips for Best Results

1. **Audio Quality**: Use high-quality recordings
2. **Monophonic**: Single instrument works best
3. **Clean Audio**: Minimize background noise
4. **File Size**: Keep under 10MB for web upload
5. **Format**: WAV or FLAC for best quality

## 🔍 Troubleshooting

### "No module named 'src'"
Make sure you're running from the project root directory.

### "MIDI file not created"
Check that the audio file contains detectable notes.

### Web app not starting
Ensure port 5000 is not in use:
```bash
# Windows
netstat -ano | findstr :5000

# macOS/Linux
lsof -i :5000
```

## 📚 Additional Resources

- **Documentation**: See README.md
- **Examples**: Check the `examples/` directory
- **Notebooks**: Explore `notebooks/` for tutorials
- **Figma**: UI/UX designs in `figma_designs/`

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Training ML Model
```bash
python src/train.py --config config.yaml
```

## 🌐 Web API Endpoints

- `GET /` - Main interface
- `POST /api/upload` - Upload audio file
- `POST /api/transcribe` - Transcribe audio
- `GET /api/download/<filename>` - Download results
- `GET /api/status` - Check API status

## 📧 Support

For issues or questions:
- Check the README.md
- Review example scripts
- Explore Jupyter notebooks

## 🎉 Next Steps

1. Try transcribing your first audio file
2. Experiment with different settings in `config.yaml`
3. Explore the Jupyter notebooks for deeper understanding
4. Train your own ML model with custom data

Happy transcribing! 🎵
