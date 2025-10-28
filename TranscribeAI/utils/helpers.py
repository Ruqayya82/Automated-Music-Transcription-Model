"""
Helper utility functions.
"""

import os
from typing import List
import mimetypes


def create_directories(dir_paths: List[str]):
    """
    Create multiple directories if they don't exist.
    
    Args:
        dir_paths: List of directory paths to create
    """
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created/verified: {dir_path}")


def validate_audio_file(file_path: str, 
                       allowed_formats: List[str] = None) -> bool:
    """
    Validate if file is a supported audio format.
    
    Args:
        file_path: Path to audio file
        allowed_formats: List of allowed file extensions
        
    Returns:
        True if valid, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac']
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower().replace('.', '')
    
    if file_ext not in allowed_formats:
        print(f"Error: Unsupported format: {file_ext}")
        print(f"Supported formats: {', '.join(allowed_formats)}")
        return False
    
    # Verify it's actually an audio file
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and not mime_type.startswith('audio'):
        print(f"Error: File is not an audio file (MIME type: {mime_type})")
        return False
    
    return True


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2:35")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size as string (e.g., "2.5 MB")
    """
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def clean_temp_files(temp_dir: str = "temp_uploads"):
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Directory containing temporary files
    """
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def ensure_output_directory(config: dict) -> str:
    """
    Ensure output directory exists.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory
    """
    output_dir = config.get('paths', {}).get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
