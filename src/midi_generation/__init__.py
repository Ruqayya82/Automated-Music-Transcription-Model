"""
MIDI Generation Module

Converts transcription results to MIDI format
"""

from .midi_creator import MIDICreator
from .note_processor import NoteProcessor

__all__ = [
    'MIDICreator',
    'NoteProcessor'
]
