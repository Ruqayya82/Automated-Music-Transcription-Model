"""
MIDI Creator Module

Creates MIDI files from note sequences
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MIDICreator:
    """Create MIDI files from note data"""
    
    def __init__(self, tempo: int = 120, velocity: int = 100):
        """
        Initialize MIDICreator
        
        Args:
            tempo: Tempo in BPM
            velocity: Default note velocity (0-127)
        """
        self.tempo = tempo
        self.velocity = velocity
        
    def create_midi(self, notes: List[Dict], 
                    output_path: str,
                    program: int = 0,
                    tempo: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI file from note list
        
        Args:
            notes: List of note dictionaries with 'start_time', 'duration', 'pitch', 'velocity'
            output_path: Path to save MIDI file
            program: MIDI program number (instrument)
            tempo: Tempo in BPM (uses default if None)
            
        Returns:
            PrettyMIDI object
        """
        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo or self.tempo)
        
        # Create instrument
        instrument = pretty_midi.Instrument(program=program)
        
        # Add notes
        for note_data in notes:
            note = pretty_midi.Note(
                velocity=note_data.get('velocity', self.velocity),
                pitch=note_data['pitch'],
                start=note_data['start_time'],
                end=note_data['start_time'] + note_data['duration']
            )
            instrument.notes.append(note)
        
        # Add instrument to MIDI
        midi.instruments.append(instrument)
        
        # Save to file
        midi.write(output_path)
        logger.info(f"Created MIDI file with {len(notes)} notes: {output_path}")
        
        return midi
    
    def create_from_pitch_onset(self, pitch_sequence: np.ndarray,
                                onset_times: np.ndarray,
                                output_path: str,
                                min_note_duration: float = 0.05) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI from pitch sequence and onset times
        
        Args:
            pitch_sequence: Array of MIDI note numbers (can contain NaN for rests)
            onset_times: Array of onset times in seconds
            output_path: Path to save MIDI file
            min_note_duration: Minimum note duration in seconds
            
        Returns:
            PrettyMIDI object
        """
        notes = []
        
        for i, onset_time in enumerate(onset_times):
            # Get pitch at onset
            pitch = pitch_sequence[i]
            
            if not np.isnan(pitch):
                # Calculate duration
                if i < len(onset_times) - 1:
                    duration = onset_times[i + 1] - onset_time
                else:
                    duration = min_note_duration * 2  # Default duration for last note
                
                # Ensure minimum duration
                duration = max(duration, min_note_duration)
                
                notes.append({
                    'start_time': onset_time,
                    'duration': duration,
                    'pitch': int(np.round(pitch)),
                    'velocity': self.velocity
                })
        
        return self.create_midi(notes, output_path)
    
    def add_tempo_changes(self, midi: pretty_midi.PrettyMIDI,
                         tempo_changes: List[tuple]):
        """
        Add tempo changes to MIDI file
        
        Args:
            midi: PrettyMIDI object
            tempo_changes: List of (time, tempo) tuples
        """
        for time, tempo in tempo_changes:
            midi.tempo_changes.append(
                pretty_midi.TempoChange(tempo, time)
            )
        logger.info(f"Added {len(tempo_changes)} tempo changes")
    
    def add_time_signature(self, midi: pretty_midi.PrettyMIDI,
                          numerator: int = 4,
                          denominator: int = 4,
                          time: float = 0.0):
        """
        Add time signature to MIDI file
        
        Args:
            midi: PrettyMIDI object
            numerator: Time signature numerator
            denominator: Time signature denominator
            time: Time to add time signature
        """
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(numerator, denominator, time)
        )
        logger.info(f"Added time signature {numerator}/{denominator}")
    
    def add_key_signature(self, midi: pretty_midi.PrettyMIDI,
                         key_number: int = 0,
                         time: float = 0.0):
        """
        Add key signature to MIDI file
        
        Args:
            midi: PrettyMIDI object
            key_number: Key signature as MIDI key number
            time: Time to add key signature
        """
        midi.key_signature_changes.append(
            pretty_midi.KeySignature(key_number, time)
        )
        logger.info(f"Added key signature: {key_number}")
    
    def quantize_notes(self, notes: List[Dict], 
                      beat_resolution: float = 0.25) -> List[Dict]:
        """
        Quantize note timing to beat grid
        
        Args:
            notes: List of note dictionaries
            beat_resolution: Beat resolution (0.25 = 16th note, 0.5 = 8th note, etc.)
            
        Returns:
            List of quantized notes
        """
        beat_duration = 60.0 / self.tempo
        grid_size = beat_duration * beat_resolution
        
        quantized_notes = []
        for note in notes:
            quantized_note = note.copy()
            
            # Quantize start time
            quantized_note['start_time'] = round(note['start_time'] / grid_size) * grid_size
            
            # Quantize duration
            quantized_note['duration'] = round(note['duration'] / grid_size) * grid_size
            quantized_note['duration'] = max(quantized_note['duration'], grid_size)
            
            quantized_notes.append(quantized_note)
        
        logger.info(f"Quantized {len(notes)} notes to {beat_resolution} beat resolution")
        return quantized_notes
    
    def merge_overlapping_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Merge overlapping notes of the same pitch
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            List of merged notes
        """
        if not notes:
            return notes
        
        # Sort notes by start time and pitch
        sorted_notes = sorted(notes, key=lambda x: (x['pitch'], x['start_time']))
        
        merged = []
        current_note = sorted_notes[0].copy()
        
        for note in sorted_notes[1:]:
            # If same pitch and overlapping/adjacent
            if (note['pitch'] == current_note['pitch'] and
                note['start_time'] <= current_note['start_time'] + current_note['duration']):
                # Extend current note
                end_time = max(
                    current_note['start_time'] + current_note['duration'],
                    note['start_time'] + note['duration']
                )
                current_note['duration'] = end_time - current_note['start_time']
            else:
                # Add current note and start new one
                merged.append(current_note)
                current_note = note.copy()
        
        merged.append(current_note)
        
        logger.info(f"Merged overlapping notes: {len(notes)} -> {len(merged)}")
        return merged
    
    def get_midi_info(self, midi_path: str) -> Dict:
        """
        Get information about a MIDI file
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary with MIDI information
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        
        info = {
            'duration': midi.get_end_time(),
            'tempo': midi.estimate_tempo(),
            'num_instruments': len(midi.instruments),
            'instruments': [],
            'total_notes': 0
        }
        
        for inst in midi.instruments:
            inst_info = {
                'name': inst.name,
                'program': inst.program,
                'is_drum': inst.is_drum,
                'num_notes': len(inst.notes)
            }
            info['instruments'].append(inst_info)
            info['total_notes'] += len(inst.notes)
        
        return info
    
    def midi_to_notes(self, midi_path: str) -> List[Dict]:
        """
        Extract notes from MIDI file
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of note dictionaries
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append({
                        'start_time': note.start,
                        'duration': note.end - note.start,
                        'pitch': note.pitch,
                        'velocity': note.velocity
                    })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start_time'])
        
        logger.info(f"Extracted {len(notes)} notes from {midi_path}")
        return notes
