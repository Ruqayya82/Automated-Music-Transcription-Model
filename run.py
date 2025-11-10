"""
TranscribeAI - Main Application Runner

This is the main entry point to run the TranscribeAI web application.
Run this file to start the server.
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from web_app.app import run_app

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 TranscribeAI - Music Transcription System")
    print("=" * 60)
    print("\nStarting web application...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    run_app(host='0.0.0.0', port=5000, debug=True)
