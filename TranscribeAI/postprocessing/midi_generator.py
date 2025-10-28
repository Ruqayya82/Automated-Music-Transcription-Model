"""
MIDI file generation from detected notes.
"""

import pretty_midi
import numpy as np
from typing import List, Tuple, Optional
import yaml


class MIDIGenerator:
    """Generate MIDI files from pitch and timing information."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize MIDIGenerator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.instrument_program = self.config['midi']['instrument']
        self.default_velocity = self.config['midi']['velocity']
        self.tempo = self.config['quantization']['tempo']
    
    def create_midi_from_notes(self, 
                               midi_notes: np.ndarray,
                               onset_times: np.ndarray,
                               durations: np.ndarray,
                               velocities: Optional[np.ndarray] = None,
                               tempo: Optional[float] = None) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI object from note information.
        
        Args:
            midi_notes: Array of MIDI note numbers
            onset_times: Array of note start times in seconds
            durations: Array of note durations in seconds
            velocities: Optional array of note velocities (0-127)
            tempo: Optional tempo in BPM
            
        Returns:
            PrettyMIDI object
        """
        if tempo is None:
            tempo = self.tempo
        
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Create an instrument
        instrument = pretty_midi.Instrument(program=self.instrument_program)
        
        # Add notes with minimum duration filter
        MIN_DURATION = 0.05  # Minimum 50ms duration to avoid MusicXML export errors
        
        for i, (note_num, start_time, duration) in enumerate(
            zip(midi_notes, onset_times, durations)
        ):
            # Skip rests (note_num = 0 or invalid)
            if note_num <= 0 or not np.isfinite(note_num):
                continue
            
            # Enforce minimum duration
            duration = max(duration, MIN_DURATION)
            
            # Get velocity
            velocity = velocities[i] if velocities is not None else self.default_velocity
            velocity = int(np.clip(velocity, 0, 127))
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note_num),
                start=start_time,
                end=start_time + duration
            )
            
            instrument.notes.append(note)
        
        # Add instrument to MIDI
        midi.instruments.append(instrument)
        
        return midi
    
    def create_midi_from_pitch_contour(self,
                                       pitch_contour: np.ndarray,
                                       times: np.ndarray,
                                       voiced_flags: np.ndarray) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI from continuous pitch contour.
        
        Args:
            pitch_contour: Array of MIDI note numbers (continuous)
            times: Time values for each pitch estimate
            voiced_flags: Boolean array indicating voiced segments
            
        Returns:
            PrettyMIDI object
        """
        # Group consecutive frames with same pitch
        notes = []
        current_note = None
        start_time = None
        
        for i, (pitch, time, voiced) in enumerate(zip(pitch_contour, times, voiced_flags)):
            if not voiced or pitch <= 0 or not np.isfinite(pitch):
                # End current note if exists
                if current_note is not None:
                    duration = time - start_time
                    if duration > 0:
                        notes.append((current_note, start_time, duration))
                    current_note = None
                continue
            
            rounded_pitch = int(round(pitch))
            
            if current_note is None:
                # Start new note
                current_note = rounded_pitch
                start_time = time
            elif rounded_pitch != current_note:
                # Pitch changed, end current note and start new one
                duration = time - start_time
                if duration > 0:
                    notes.append((current_note, start_time, duration))
                current_note = rounded_pitch
                start_time = time
        
        # Add final note
        if current_note is not None and len(times) > 0:
            duration = times[-1] - start_time
            if duration > 0:
                notes.append((current_note, start_time, duration))
        
        # Convert to arrays
        if len(notes) > 0:
            midi_notes, onset_times, durations = zip(*notes)
            return self.create_midi_from_notes(
                np.array(midi_notes),
                np.array(onset_times),
                np.array(durations)
            )
        else:
            # Return empty MIDI
            return pretty_midi.PrettyMIDI()
    
    def save_midi(self, midi: pretty_midi.PrettyMIDI, output_path: str):
        """
        Save MIDI object to file.
        
        Args:
            midi: PrettyMIDI object
            output_path: Output file path
        """
        midi.write(output_path)
        print(f"MIDI file saved to: {output_path}")
    
    def add_tempo_changes(self, midi: pretty_midi.PrettyMIDI, 
                         tempo_changes: List[Tuple[float, float]]):
        """
        Add tempo changes to MIDI file.
        
        Args:
            midi: PrettyMIDI object
            tempo_changes: List of (time, tempo) tuples
        """
        for time, tempo in tempo_changes:
            midi.tempo_changes.append(
                pretty_midi.TempoChange(tempo=tempo, time=time)
            )
    
    def quantize_timing(self, 
                       onset_times: np.ndarray,
                       durations: np.ndarray,
                       tempo: float = None,
                       subdivision: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize note timings to rhythmic grid.
        
        Args:
            onset_times: Original onset times
            durations: Original durations
            tempo: Tempo in BPM
            subdivision: Subdivision (16 = sixteenth notes)
            
        Returns:
            Tuple of (quantized_onsets, quantized_durations)
        """
        if tempo is None:
            tempo = self.tempo
        
        # Calculate grid spacing
        beat_duration = 60.0 / tempo
        grid_spacing = beat_duration / (subdivision / 4)
        
        # Quantize onsets
        quantized_onsets = np.round(onset_times / grid_spacing) * grid_spacing
        
        # Quantize durations
        quantized_durations = np.maximum(
            np.round(durations / grid_spacing) * grid_spacing,
            grid_spacing  # Minimum duration is one grid unit
        )
        
        return quantized_onsets, quantized_durations
    
    def estimate_tempo(self, onset_times: np.ndarray) -> float:
        """
        Estimate tempo from onset times.
        
        Args:
            onset_times: Array of onset times
            
        Returns:
            Estimated tempo in BPM
        """
        if len(onset_times) < 2:
            return self.tempo
        
        # Calculate inter-onset intervals
        intervals = np.diff(onset_times)
        
        # Estimate beat duration as median interval
        beat_duration = np.median(intervals)
        
        # Convert to BPM
        tempo = 60.0 / beat_duration
        
        # Clamp to reasonable range
        tempo = np.clip(tempo, 40, 240)
        
        return tempo
    
    def get_midi_info(self, midi: pretty_midi.PrettyMIDI) -> dict:
        """
        Extract information from MIDI file.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            Dictionary with MIDI information
        """
        info = {
            'duration': midi.get_end_time(),
            'tempo': midi.estimate_tempo(),
            'num_instruments': len(midi.instruments),
            'total_notes': sum(len(inst.notes) for inst in midi.instruments),
            'time_signature_changes': len(midi.time_signature_changes),
            'key_signature_changes': len(midi.key_signature_changes)
        }
        
        return info
