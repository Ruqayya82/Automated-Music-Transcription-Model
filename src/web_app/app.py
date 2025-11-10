"""
Flask Web Application

Main web application for music transcription
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import yaml
import logging
from pathlib import Path
from typing import Optional

from src.models.transcription_model import TranscriptionModel
from src.midi_generation import MIDICreator
from src.musicxml_conversion import MusicXMLGenerator

logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)
    
    # Get project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Load configuration
    if config_path is None:
        config_path = project_root / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure app with absolute paths
    app.config['MAX_CONTENT_LENGTH'] = config['web']['max_file_size']
    app.config['UPLOAD_FOLDER'] = str(project_root / config['web']['upload_folder'])
    app.config['OUTPUT_FOLDER'] = str(project_root / config['web']['output_folder'])
    app.config['ALLOWED_EXTENSIONS'] = set(config['web']['allowed_extensions'])
    app.config['CONFIG'] = config
    
    # Create folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Initialize models (lazy loading)
    app.transcription_model = None
    app.midi_creator = None
    app.musicxml_generator = None
    
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Upload audio file"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Uploaded file: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    @app.route('/api/transcribe', methods=['POST'])
    def transcribe():
        """Transcribe audio file"""
        data = request.json
        filename = data.get('filename')
        use_model = data.get('use_model', False)
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        try:
            # Initialize model if needed
            if app.transcription_model is None:
                app.transcription_model = TranscriptionModel(
                    config=app.config['CONFIG'],
                    device='cpu'
                )
            
            # Transcribe
            notes, metadata = app.transcription_model.transcribe(
                filepath,
                use_model=use_model
            )
            
            # Generate MIDI using PrettyMIDI
            if app.midi_creator is None:
                app.midi_creator = MIDICreator(
                    tempo=app.config['CONFIG']['midi']['tempo'],
                    velocity=app.config['CONFIG']['midi']['velocity']
                )
            
            output_name = os.path.splitext(filename)[0]
            midi_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_name}.mid')
            
            # Create MIDI file from transcribed notes (not audio conversion!)
            midi_obj = app.midi_creator.create_midi(notes, midi_path)
            
            # Validate MIDI file was created properly
            if not os.path.exists(midi_path):
                raise Exception("MIDI file generation failed")
            
            # Get MIDI file info for metadata
            midi_info = app.midi_creator.get_midi_info(midi_path)
            logger.info(f"Generated MIDI with {midi_info['total_notes']} notes, duration: {midi_info['duration']:.2f}s")
            
            # Generate MusicXML
            if app.musicxml_generator is None:
                musicxml_config = app.config['CONFIG']['musicxml']
                app.musicxml_generator = MusicXMLGenerator(
                    title=musicxml_config['title'],
                    composer=musicxml_config['composer'],
                    time_signature=musicxml_config['time_signature'],
                    key_signature=musicxml_config['key_signature'],
                    clef_type=musicxml_config['clef']
                )
            
            xml_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{output_name}.musicxml')
            app.musicxml_generator.midi_to_musicxml(midi_path, xml_path)
            
            logger.info(f"Transcription complete: {len(notes)} notes detected from audio")
            
            return jsonify({
                'success': True,
                'num_notes': len(notes),
                'midi_file': f'{output_name}.mid',
                'musicxml_file': f'{output_name}.musicxml',
                'metadata': {
                    **metadata,
                    'midi_info': {
                        'duration': midi_info['duration'],
                        'total_notes': midi_info['total_notes'],
                        'tempo': midi_info['tempo']
                    }
                },
                'processing_time': metadata.get('processing_time', 0)
            })
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/download/<filename>')
    def download_file(filename):
        """Download generated file"""
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
    
    @app.route('/api/status')
    def status():
        """API status check"""
        return jsonify({
            'status': 'online',
            'model_loaded': app.transcription_model is not None
        })
    
    @app.errorhandler(413)
    def too_large(e):
        """Handle file too large error"""
        return jsonify({'error': 'File too large'}), 413
    
    return app


def run_app(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """
    Run the Flask application
    
    Args:
        host: Host address
        port: Port number
        debug: Debug mode
    """
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
