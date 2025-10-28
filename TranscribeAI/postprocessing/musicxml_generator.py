"""
MusicXML file generation from MIDI for sheet music display.
"""

from music21 import stream, note, tempo, meter, metadata, instrument
import pretty_midi
import numpy as np
from typing import Optional
import yaml


class MusicXMLGenerator:
    """Generate MusicXML files from MIDI for printable sheet music."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize MusicXMLGenerator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tempo = self.config['quantization']['tempo']
        self.time_signature = self.config['quantization']['time_signature']
    
    def midi_to_musicxml(self, 
                        midi_path: str,
                        title: str = "Transcribed Music",
                        composer: str = "TranscribeAI") -> stream.Score:
        """
        Convert MIDI file to MusicXML Score.
        
        Args:
            midi_path: Path to MIDI file
            title: Title for the score
            composer: Composer name for the score
            
        Returns:
            music21 Score object
        """
        # Parse MIDI file
        score = stream.Score()
        
        # Add metadata
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer
        
        # Parse MIDI and convert to music21
        try:
            midi_stream = stream.Score()
            midi_stream.insert(0, stream.converter.parse(midi_path))
            
            # Extract parts from MIDI
            for part in midi_stream.parts:
                score.append(part)
        except:
            # Fallback: manually create from MIDI
            score = self._create_score_from_midi(midi_path, title, composer)
        
        return score
    
    def _create_score_from_midi(self,
                                midi_path: str,
                                title: str,
                                composer: str) -> stream.Score:
        """
        Manually create music21 Score from MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            title: Title for the score
            composer: Composer name
            
        Returns:
            music21 Score object
        """
        # Load MIDI
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Create score
        score = stream.Score()
        
        # Add metadata
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer
        
        # Create part for each instrument
        for midi_inst in midi_data.instruments:
            part = stream.Part()
            
            # Set instrument
            m21_instrument = instrument.Piano()  # Default to piano
            part.append(m21_instrument)
            
            # Add tempo
            tempo_marking = tempo.MetronomeMark(number=self.tempo)
            part.append(tempo_marking)
            
            # Add time signature
            time_sig = meter.TimeSignature(self.time_signature)
            part.append(time_sig)
            
            # Add notes
            for midi_note in midi_inst.notes:
                # Create music21 note
                duration_quarters = (midi_note.end - midi_note.start) * (self.tempo / 60.0)
                
                m21_note = note.Note(midi_note.pitch)
                m21_note.quarterLength = duration_quarters
                m21_note.volume.velocity = midi_note.velocity
                
                # Calculate offset in quarter notes
                offset_quarters = midi_note.start * (self.tempo / 60.0)
                part.insert(offset_quarters, m21_note)
            
            score.append(part)
        
        return score
    
    def create_score_from_notes(self,
                                midi_notes: np.ndarray,
                                durations: np.ndarray,
                                title: str = "Transcribed Music",
                                composer: str = "TranscribeAI") -> stream.Score:
        """
        Create MusicXML Score directly from note arrays.
        
        Args:
            midi_notes: Array of MIDI note numbers
            durations: Array of note durations in seconds
            title: Title for the score
            composer: Composer name
            
        Returns:
            music21 Score object
        """
        # Create score
        score = stream.Score()
        
        # Add metadata
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer
        
        # Create part
        part = stream.Part()
        
        # Set instrument (Piano)
        part.append(instrument.Piano())
        
        # Add tempo
        tempo_marking = tempo.MetronomeMark(number=self.tempo)
        part.append(tempo_marking)
        
        # Add time signature
        time_sig = meter.TimeSignature(self.time_signature)
        part.append(time_sig)
        
        # Add notes with minimum duration filter
        current_offset = 0.0
        MIN_QUARTER_LENGTH = 0.125  # Minimum 32nd note to avoid MusicXML export errors
        
        for midi_note, duration in zip(midi_notes, durations):
            if midi_note > 0 and np.isfinite(midi_note):
                # Convert duration from seconds to quarter notes
                duration_quarters = duration * (self.tempo / 60.0)
                
                # Enforce minimum duration to avoid MusicXML errors
                duration_quarters = max(duration_quarters, MIN_QUARTER_LENGTH)
                
                # Create note
                m21_note = note.Note(int(midi_note))
                m21_note.quarterLength = duration_quarters
                
                part.insert(current_offset, m21_note)
                current_offset += duration_quarters
            else:
                # Rest
                duration_quarters = duration * (self.tempo / 60.0)
                
                # Enforce minimum duration for rests too
                duration_quarters = max(duration_quarters, MIN_QUARTER_LENGTH)
                
                m21_rest = note.Rest()
                m21_rest.quarterLength = duration_quarters
                
                part.insert(current_offset, m21_rest)
                current_offset += duration_quarters
        
        score.append(part)
        return score
    
    def save_musicxml(self, score: stream.Score, output_path: str):
        """
        Save Score to MusicXML file.
        
        Args:
            score: music21 Score object
            output_path: Output file path
        """
        score.write('musicxml', fp=output_path)
        print(f"MusicXML file saved to: {output_path}")
    
    def save_as_pdf(self, score: stream.Score, output_path: str):
        """
        Save Score as PDF (requires MuseScore or similar).
        
        Args:
            score: music21 Score object
            output_path: Output file path
        """
        try:
            score.write('lily.pdf', fp=output_path)
            print(f"PDF file saved to: {output_path}")
        except:
            print("PDF generation requires MuseScore or LilyPond to be installed.")
            print("Saving as MusicXML instead.")
            xml_path = output_path.replace('.pdf', '.xml')
            self.save_musicxml(score, xml_path)
    
    def save_as_png(self, score: stream.Score, output_path: str):
        """
        Save Score as PNG image (requires MuseScore).
        
        Args:
            score: music21 Score object
            output_path: Output file path
        """
        try:
            score.write('lily.png', fp=output_path)
            print(f"PNG file saved to: {output_path}")
        except:
            print("PNG generation requires MuseScore or LilyPond to be installed.")
    
    def add_lyrics(self, score: stream.Score, lyrics: list):
        """
        Add lyrics to the score.
        
        Args:
            score: music21 Score object
            lyrics: List of lyric strings
        """
        for part in score.parts:
            notes_list = list(part.flatten().notes)
            for i, lyric in enumerate(lyrics):
                if i < len(notes_list):
                    notes_list[i].addLyric(lyric)
    
    def analyze_key(self, score: stream.Score) -> str:
        """
        Analyze and determine the key of the score.
        
        Args:
            score: music21 Score object
            
        Returns:
            Key signature as string
        """
        key = score.analyze('key')
        return str(key)
    
    def simplify_notation(self, score: stream.Score) -> stream.Score:
        """
        Simplify notation by quantizing to standard note values.
        
        Args:
            score: music21 Score object
            
        Returns:
            Simplified Score object
        """
        # This would implement quantization to standard note values
        # (whole, half, quarter, eighth, etc.)
        for part in score.parts:
            for n in part.flatten().notesAndRests:
                # Round to nearest standard duration
                ql = n.quarterLength
                if ql >= 3.5:
                    n.quarterLength = 4.0  # Whole note
                elif ql >= 1.75:
                    n.quarterLength = 2.0  # Half note
                elif ql >= 0.875:
                    n.quarterLength = 1.0  # Quarter note
                elif ql >= 0.4375:
                    n.quarterLength = 0.5  # Eighth note
                else:
                    n.quarterLength = 0.25  # Sixteenth note
        
        return score
