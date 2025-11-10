"""
MusicXML Generator Module

Converts MIDI files to MusicXML for sheet music visualization
"""

from music21 import converter, stream, note, clef, tempo, key, meter, chord
from music21.musicxml import m21ToXml
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class MusicXMLGenerator:
    """Generate MusicXML from MIDI files"""
    
    def __init__(self, title: str = "Transcribed Score",
                 composer: str = "AI Transcription",
                 time_signature: str = "4/4",
                 key_signature: str = "C",
                 clef_type: str = "treble"):
        """
        Initialize MusicXMLGenerator
        
        Args:
            title: Score title
            composer: Composer name
            time_signature: Time signature (e.g., "4/4", "3/4")
            key_signature: Key signature (e.g., "C", "G", "Am")
            clef_type: Clef type ("treble", "bass", "alto", "tenor")
        """
        self.title = title
        self.composer = composer
        self.time_signature = time_signature
        self.key_signature = key_signature
        self.clef_type = clef_type
        
    def midi_to_musicxml(self, midi_path: str, 
                        output_path: str,
                        add_metadata: bool = True) -> stream.Score:
        """
        Convert MIDI file to MusicXML
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path to save MusicXML file
            add_metadata: Add title, composer, etc. if True
            
        Returns:
            Music21 Score object
        """
        # Load MIDI file
        score = converter.parse(midi_path)
        
        # Add metadata
        if add_metadata:
            score.metadata.title = self.title
            score.metadata.composer = self.composer
        
        # Process the score
        score = self._process_score(score)
        
        # Write MusicXML
        score.write('musicxml', fp=output_path)
        logger.info(f"Created MusicXML file: {output_path}")
        
        return score
    
    def _process_score(self, score: stream.Score) -> stream.Score:
        """
        Process and clean up the score
        
        Args:
            score: Music21 score object
            
        Returns:
            Processed score
        """
        # Get or create parts
        if len(score.parts) == 0:
            # Create a new part if none exist
            part = stream.Part()
            score.insert(0, part)
        
        for part in score.parts:
            # Add clef if not present
            if not part.recurse().getElementsByClass(clef.Clef):
                part.insert(0, self._get_clef())
            
            # Add time signature if not present
            if not part.recurse().getElementsByClass(meter.TimeSignature):
                numerator, denominator = self.time_signature.split('/')
                part.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
            
            # Add key signature if not present
            if not part.recurse().getElementsByClass(key.KeySignature):
                part.insert(0, key.Key(self.key_signature))
        
        return score
    
    def _get_clef(self) -> clef.Clef:
        """Get appropriate clef object"""
        clef_map = {
            'treble': clef.TrebleClef(),
            'bass': clef.BassClef(),
            'alto': clef.AltoClef(),
            'tenor': clef.TenorClef()
        }
        return clef_map.get(self.clef_type, clef.TrebleClef())
    
    def create_from_notes(self, notes_list: list,
                         output_path: str,
                         tempo_bpm: int = 120) -> stream.Score:
        """
        Create MusicXML from note list
        
        Args:
            notes_list: List of note dictionaries
            output_path: Path to save MusicXML
            tempo_bpm: Tempo in BPM
            
        Returns:
            Music21 Score object
        """
        # Create score
        score = stream.Score()
        score.metadata.title = self.title
        score.metadata.composer = self.composer
        
        # Create part
        part = stream.Part()
        
        # Add tempo
        part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
        
        # Add clef
        part.insert(0, self._get_clef())
        
        # Add time signature
        numerator, denominator = self.time_signature.split('/')
        part.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
        
        # Add key signature
        part.insert(0, key.Key(self.key_signature))
        
        # Sort notes by start time
        sorted_notes = sorted(notes_list, key=lambda x: x['start_time'])
        
        # Convert notes to music21 notes
        for note_data in sorted_notes:
            pitch_num = note_data['pitch']
            duration_seconds = note_data['duration']
            
            # Create note
            n = note.Note(pitch_num)
            n.quarterLength = self._seconds_to_quarter_length(
                duration_seconds, tempo_bpm
            )
            
            # Set dynamics
            velocity = note_data.get('velocity', 80)
            n.volume.velocity = velocity
            
            part.append(n)
        
        score.insert(0, part)
        
        # Write to file
        score.write('musicxml', fp=output_path)
        logger.info(f"Created MusicXML from {len(notes_list)} notes: {output_path}")
        
        return score
    
    def _seconds_to_quarter_length(self, seconds: float, tempo: int) -> float:
        """
        Convert duration in seconds to quarter note length
        
        Args:
            seconds: Duration in seconds
            tempo: Tempo in BPM
            
        Returns:
            Quarter note length
        """
        quarter_duration = 60.0 / tempo
        return seconds / quarter_duration
    
    def add_lyrics(self, score: stream.Score, lyrics_text: str) -> stream.Score:
        """
        Add lyrics to score
        
        Args:
            score: Music21 score
            lyrics_text: Lyrics text (space-separated syllables)
            
        Returns:
            Score with lyrics
        """
        syllables = lyrics_text.split()
        
        for part in score.parts:
            notes_list = list(part.recurse().notes)
            for i, n in enumerate(notes_list):
                if i < len(syllables):
                    n.lyric = syllables[i]
        
        logger.info(f"Added {len(syllables)} lyric syllables")
        return score
    
    def simplify_notation(self, score: stream.Score) -> stream.Score:
        """
        Simplify notation for easier reading
        
        Args:
            score: Music21 score
            
        Returns:
            Simplified score
        """
        for part in score.parts:
            # Quantize durations to common note values
            part.quantize([1.0, 0.5, 0.25, 0.125], inPlace=True)
        
        logger.info("Simplified notation")
        return score
    
    def transpose_score(self, score: stream.Score, 
                       semitones: int) -> stream.Score:
        """
        Transpose entire score
        
        Args:
            score: Music21 score
            semitones: Number of semitones to transpose
            
        Returns:
            Transposed score
        """
        transposed = score.transpose(semitones)
        logger.info(f"Transposed score by {semitones} semitones")
        return transposed
    
    def export_to_pdf(self, score: stream.Score, output_path: str):
        """
        Export score to PDF (requires MuseScore or Lilypond)
        
        Args:
            score: Music21 score
            output_path: Path to save PDF
        """
        try:
            score.write('lily.pdf', fp=output_path)
            logger.info(f"Exported PDF: {output_path}")
        except Exception as e:
            logger.warning(f"Could not export to PDF: {str(e)}")
            logger.info("Install MuseScore or Lilypond for PDF export")
    
    def export_to_png(self, score: stream.Score, output_path: str):
        """
        Export score to PNG image (requires MuseScore or Lilypond)
        
        Args:
            score: Music21 score
            output_path: Path to save PNG
        """
        try:
            score.write('lily.png', fp=output_path)
            logger.info(f"Exported PNG: {output_path}")
        except Exception as e:
            logger.warning(f"Could not export to PNG: {str(e)}")
            logger.info("Install MuseScore or Lilypond for PNG export")
    
    def get_score_info(self, score: stream.Score) -> Dict:
        """
        Get information about a score
        
        Args:
            score: Music21 score
            
        Returns:
            Dictionary with score information
        """
        info = {
            'title': score.metadata.title if score.metadata else None,
            'composer': score.metadata.composer if score.metadata else None,
            'duration': score.duration.quarterLength,
            'num_parts': len(score.parts),
            'num_measures': 0,
            'num_notes': 0
        }
        
        for part in score.parts:
            measures = part.getElementsByClass(stream.Measure)
            info['num_measures'] = max(info['num_measures'], len(measures))
            info['num_notes'] += len(list(part.recurse().notes))
        
        return info
    
    def analyze_score(self, score: stream.Score) -> Dict:
        """
        Analyze score for key, tempo, and other characteristics
        
        Args:
            score: Music21 score
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Analyze key
        try:
            analyzed_key = score.analyze('key')
            analysis['key'] = str(analyzed_key)
        except:
            analysis['key'] = 'Unknown'
        
        # Get tempo
        tempo_marks = score.recurse().getElementsByClass(tempo.MetronomeMark)
        if tempo_marks:
            analysis['tempo'] = tempo_marks[0].number
        else:
            analysis['tempo'] = None
        
        # Get time signature
        time_sigs = score.recurse().getElementsByClass(meter.TimeSignature)
        if time_sigs:
            analysis['time_signature'] = str(time_sigs[0])
        else:
            analysis['time_signature'] = None
        
        # Count notes
        analysis['total_notes'] = len(list(score.recurse().notes))
        
        logger.info(f"Analyzed score: {analysis}")
        return analysis
    
    def validate_musicxml(self, musicxml_path: str) -> bool:
        """
        Validate MusicXML file
        
        Args:
            musicxml_path: Path to MusicXML file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            score = converter.parse(musicxml_path)
            logger.info(f"MusicXML file is valid: {musicxml_path}")
            return True
        except Exception as e:
            logger.error(f"Invalid MusicXML file: {str(e)}")
            return False
