"""
Note Processor Module

Processes and refines note sequences
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class NoteProcessor:
    """Process and refine note sequences"""
    
    def __init__(self, min_duration: float = 0.05):
        """
        Initialize NoteProcessor
        
        Args:
            min_duration: Minimum note duration in seconds
        """
        self.min_duration = min_duration
        
    def filter_short_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Remove notes shorter than minimum duration
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Filtered notes
        """
        filtered = [
            note for note in notes 
            if note['duration'] >= self.min_duration
        ]
        logger.info(f"Filtered short notes: {len(notes)} -> {len(filtered)}")
        return filtered
    
    def remove_duplicate_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Remove duplicate notes (same pitch, start time, duration)
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            De-duplicated notes
        """
        seen = set()
        unique = []
        
        for note in notes:
            key = (note['pitch'], note['start_time'], note['duration'])
            if key not in seen:
                seen.add(key)
                unique.append(note)
        
        logger.info(f"Removed duplicates: {len(notes)} -> {len(unique)}")
        return unique
    
    def split_overlapping_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Split overlapping notes of the same pitch into separate notes
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Split notes
        """
        if not notes:
            return notes
        
        # Sort by pitch and start time
        sorted_notes = sorted(notes, key=lambda x: (x['pitch'], x['start_time']))
        
        result = []
        for note in sorted_notes:
            # Check for overlap with last note of same pitch
            overlaps = [n for n in result 
                       if n['pitch'] == note['pitch'] and
                       n['start_time'] + n['duration'] > note['start_time']]
            
            if overlaps:
                # Adjust previous note duration
                prev_note = overlaps[-1]
                prev_note['duration'] = note['start_time'] - prev_note['start_time']
            
            result.append(note.copy())
        
        logger.info(f"Split overlapping notes")
        return result
    
    def adjust_note_durations(self, notes: List[Dict],
                             max_gap: float = 0.1) -> List[Dict]:
        """
        Adjust note durations to fill small gaps
        
        Args:
            notes: List of note dictionaries
            max_gap: Maximum gap to fill (seconds)
            
        Returns:
            Notes with adjusted durations
        """
        if not notes:
            return notes
        
        # Sort by start time
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])
        
        adjusted = []
        for i, note in enumerate(sorted_notes):
            adjusted_note = note.copy()
            
            # Check if there's a next note
            if i < len(sorted_notes) - 1:
                next_note = sorted_notes[i + 1]
                gap = next_note['start_time'] - (note['start_time'] + note['duration'])
                
                # If gap is small, extend note to next note
                if 0 < gap <= max_gap:
                    adjusted_note['duration'] = next_note['start_time'] - note['start_time']
            
            adjusted.append(adjusted_note)
        
        logger.info(f"Adjusted note durations to fill gaps")
        return adjusted
    
    def transpose_notes(self, notes: List[Dict], semitones: int) -> List[Dict]:
        """
        Transpose all notes by semitones
        
        Args:
            notes: List of note dictionaries
            semitones: Number of semitones to transpose (positive or negative)
            
        Returns:
            Transposed notes
        """
        transposed = []
        for note in notes:
            transposed_note = note.copy()
            new_pitch = note['pitch'] + semitones
            
            # Ensure pitch is in valid MIDI range (0-127)
            if 0 <= new_pitch <= 127:
                transposed_note['pitch'] = new_pitch
                transposed.append(transposed_note)
        
        logger.info(f"Transposed {len(notes)} notes by {semitones} semitones")
        return transposed
    
    def normalize_velocities(self, notes: List[Dict],
                            target_velocity: int = 80) -> List[Dict]:
        """
        Normalize all note velocities
        
        Args:
            notes: List of note dictionaries
            target_velocity: Target velocity value
            
        Returns:
            Notes with normalized velocities
        """
        normalized = []
        for note in notes:
            normalized_note = note.copy()
            normalized_note['velocity'] = target_velocity
            normalized.append(normalized_note)
        
        return normalized
    
    def apply_dynamics(self, notes: List[Dict],
                      dynamic_curve: str = 'linear') -> List[Dict]:
        """
        Apply dynamic curve to note velocities
        
        Args:
            notes: List of note dictionaries
            dynamic_curve: Type of curve ('linear', 'crescendo', 'diminuendo')
            
        Returns:
            Notes with dynamic curve applied
        """
        if not notes:
            return notes
        
        result = []
        for i, note in enumerate(notes):
            note_copy = note.copy()
            progress = i / max(len(notes) - 1, 1)
            
            if dynamic_curve == 'crescendo':
                # Increase velocity over time
                velocity = int(60 + 60 * progress)
            elif dynamic_curve == 'diminuendo':
                # Decrease velocity over time
                velocity = int(120 - 60 * progress)
            else:  # linear or default
                velocity = note.get('velocity', 80)
            
            note_copy['velocity'] = max(0, min(127, velocity))
            result.append(note_copy)
        
        logger.info(f"Applied {dynamic_curve} dynamic curve")
        return result
    
    def get_note_statistics(self, notes: List[Dict]) -> Dict:
        """
        Calculate statistics about note sequence
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Dictionary of statistics
        """
        if not notes:
            return {
                'num_notes': 0,
                'total_duration': 0,
                'pitch_range': (0, 0),
                'avg_duration': 0,
                'avg_velocity': 0
            }
        
        pitches = [n['pitch'] for n in notes]
        durations = [n['duration'] for n in notes]
        velocities = [n.get('velocity', 80) for n in notes]
        
        # Calculate span of time
        start_times = [n['start_time'] for n in notes]
        end_times = [n['start_time'] + n['duration'] for n in notes]
        total_duration = max(end_times) - min(start_times)
        
        stats = {
            'num_notes': len(notes),
            'total_duration': total_duration,
            'pitch_range': (min(pitches), max(pitches)),
            'pitch_mean': np.mean(pitches),
            'avg_duration': np.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_velocity': np.mean(velocities),
            'notes_per_second': len(notes) / total_duration if total_duration > 0 else 0
        }
        
        return stats
    
    def detect_tempo_from_notes(self, notes: List[Dict]) -> float:
        """
        Estimate tempo from note inter-onset intervals
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Estimated tempo in BPM
        """
        if len(notes) < 2:
            return 120.0  # Default tempo
        
        # Sort by start time
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])
        
        # Calculate inter-onset intervals
        intervals = []
        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i + 1]['start_time'] - sorted_notes[i]['start_time']
            if interval > 0:
                intervals.append(interval)
        
        if not intervals:
            return 120.0
        
        # Use median interval as beat duration
        median_interval = np.median(intervals)
        
        # Convert to BPM
        tempo = 60.0 / median_interval
        
        # Clamp to reasonable range
        tempo = max(40, min(240, tempo))
        
        logger.info(f"Estimated tempo: {tempo:.1f} BPM")
        return tempo
    
    def align_to_grid(self, notes: List[Dict], 
                     grid_size: float = 0.125) -> List[Dict]:
        """
        Align note start times and durations to grid
        
        Args:
            notes: List of note dictionaries
            grid_size: Grid size in seconds
            
        Returns:
            Grid-aligned notes
        """
        aligned = []
        for note in notes:
            aligned_note = note.copy()
            
            # Align start time
            aligned_note['start_time'] = round(note['start_time'] / grid_size) * grid_size
            
            # Align duration
            aligned_note['duration'] = max(
                grid_size,
                round(note['duration'] / grid_size) * grid_size
            )
            
            aligned.append(aligned_note)
        
        logger.info(f"Aligned {len(notes)} notes to grid size {grid_size}s")
        return aligned
    
    def extract_melody(self, notes: List[Dict]) -> List[Dict]:
        """
        Extract melody (highest notes) from polyphonic sequence
        
        Args:
            notes: List of note dictionaries
            
        Returns:
            Melody notes
        """
        if not notes:
            return notes
        
        # Group notes by overlapping time windows
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])
        melody = []
        
        i = 0
        while i < len(sorted_notes):
            current_time = sorted_notes[i]['start_time']
            
            # Find all notes that overlap with current time
            overlapping = []
            for note in sorted_notes[i:]:
                if note['start_time'] <= current_time < note['start_time'] + note['duration']:
                    overlapping.append(note)
                elif note['start_time'] > current_time:
                    break
            
            if overlapping:
                # Select highest pitch
                highest = max(overlapping, key=lambda x: x['pitch'])
                melody.append(highest)
                i += len(overlapping)
            else:
                i += 1
        
        logger.info(f"Extracted melody: {len(notes)} -> {len(melody)} notes")
        return melody
