import streamlit as st
import os
import tempfile
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21
from streamlit_music_score import music_score
import streamlit.components.v1 as components
import verovio

# --- 1. Setup & Config ---
st.set_page_config(page_title="AI Music Transcriber", layout="wide")
st.title("🎵 Audio to Sheet Music Converter")
st.markdown("""
**Instructions:**
1. Select your instrument.
2. Upload a clear, monophonic audio file.
3. **Tip:** Uncheck "Simplify Rhythms" if you specifically need fast 16th notes.
""")

# --- 2. Constants & Theory ---
INSTRUMENT_SETTINGS = {
    "Piano (C)": {"semitones": 0, "single_note": False},
    "Guitar (C)": {"semitones": 0, "single_note": False},
    "Violin (C)": {"semitones": 0, "single_note": False},
    "Flute (C)": {"semitones": 0, "single_note": True},
    "Trumpet (Bb)": {"semitones": 2, "single_note": True},
    "Clarinet (Bb)": {"semitones": 2, "single_note": True},
    "Alto Sax (Eb)": {"semitones": 9, "single_note": True},
    "Tenor Sax (Bb)": {"semitones": 2, "single_note": True},
    "French Horn (F)": {"semitones": 7, "single_note": True},
}

# --- 3. Helper Functions ---

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def audio_to_midi(audio_path):
    try:
        model_output, midi_data, note_events = predict(
            audio_path, 
            ICASSP_2022_MODEL_PATH, 
            onset_threshold=0.6,   
            frame_threshold=0.4,
            minimum_note_length=100.0, 
        )
        return midi_data
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def clean_engraving(score, grid_size=0.25):
    """
    Crucial function for readable sheet music.
    1. Flattens chords to single notes (if monophonic context).
    2. Removes Overlaps (Voice 1 vs Voice 2 issues).
    3. Fills Gaps (removes tiny rests).
    4. Snaps to Grid (Quantization).
    """
    # Extract all notes and sort them by start time
    all_notes = []
    
    # Flatten and grab notes/chords
    for element in score.flat.notes:
        # Convert Chords to single Note (Top pitch)
        if isinstance(element, music21.chord.Chord):
            pitch = element.sortAscending().pitches[-1]
            n = music21.note.Note(pitch)
            n.offset = element.offset
            n.duration = element.duration
            all_notes.append(n)
        else:
            all_notes.append(element)

    if not all_notes:
        return score

    # Sort strictly by offset
    all_notes.sort(key=lambda x: x.offset)
    
    # Create a new clean stream
    clean_stream = music21.stream.Score()
    part = music21.stream.Part()
    
    for i in range(len(all_notes)):
        current_n = all_notes[i]
        
        # 1. QUANTIZE START TIME
        # Snap offset to nearest grid_size (0.25 = 16th note)
        quantized_offset = round(current_n.offset / grid_size) * grid_size
        current_n.offset = quantized_offset
        
        # Handle Last Note
        if i == len(all_notes) - 1:
            # Quantize duration too
            q_dur = round(current_n.quarterLength / grid_size) * grid_size
            current_n.quarterLength = max(grid_size, q_dur)
            part.insert(current_n.offset, current_n)
            break

        next_n = all_notes[i+1]
        next_offset_raw = next_n.offset
        # Look ahead to see where next note *will* be quantized
        next_offset_quant = round(next_offset_raw / grid_size) * grid_size
        
        # 2. FIX OVERLAPS & GAPS
        # Force current note to end exactly where next note starts
        # This creates a perfect "Legato" line with no rests and no overlaps.
        
        # Calculate gap
        gap = next_offset_quant - current_n.offset
        
        if gap <= 0:
            # If quantization made them start at same time (or before), 
            # give it a minimum duration and push next note? 
            # For simplicity, we skip extremely short notes that got crushed.
            continue 
        
        # 3. GAP FILLING LOGIC
        # If the note naturally ends way before the next one, keep the rest.
        # But if it ends close (within 1 beat), extend it to touch the next note.
        natural_end = current_n.offset + current_n.quarterLength
        
        if (next_offset_quant - natural_end) < 1.0: # If gap is < 1 beat
            # Extend to fill the gap (Legato)
            current_n.quarterLength = gap
        else:
            # Keep the natural duration (quantized) but ensure no overlap
            q_dur = round(current_n.quarterLength / grid_size) * grid_size
            current_n.quarterLength = min(q_dur, gap)

        part.insert(current_n.offset, current_n)

    clean_stream.insert(0, part)
    return clean_stream

def midi_to_musicxml(midi_data, output_xml_path, transposition_semitones=0, force_single_note=False, simplify_rhythm=True):
    try:
        temp_midi_path = output_xml_path.replace(".musicxml", ".mid")
        midi_data.write(temp_midi_path)
        
        score = music21.converter.parse(temp_midi_path)
        
        if transposition_semitones != 0:
            interval = music21.interval.Interval(transposition_semitones)
            score = score.transpose(interval)

        # --- NEW CLEANUP PIPELINE ---
        
        # Determine Grid Size
        # simplify_rhythm=True -> Snap to Eighths (0.5)
        # simplify_rhythm=False -> Snap to Sixteenths (0.25)
        grid_size = 0.5 if simplify_rhythm else 0.25
        
        if force_single_note:
            # This function replaces the old "enforce_monophony"
            # It rebuilds the stream completely linearly
            score = clean_engraving(score, grid_size=grid_size)
        
        # Standard Setup
        if score.parts:
            p = score.parts[0]
        else:
            p = score
            
        # FIX: Insert directly into Stream/Part at offset 0
        # Do NOT search for p.measure(1) because it might not exist yet.
        p.insert(0, music21.meter.TimeSignature('4/4'))
        
        # Analyze key from the flat stream
        key = p.analyze('key')
        p.insert(0, key)
        
        # Final Polish (makeNotation will create the measures for us)
        score.makeNotation(inPlace=True)
        score.stripTies(inPlace=True)
        
        score.write('musicxml', fp=output_xml_path)
        return True
    except Exception as e:
        st.error(f"MusicXML Error: {e}")
        # raise e # Uncomment for debugging
        return False

def render_svg(musicxml_path):
    try:
        tk = verovio.toolkit()
        tk.loadFile(musicxml_path)
        tk.setOptions({
            "pageWidth": 2100,
            "adjustPageHeight": True,
            "scale": 40,
            "font": "Bravura"
        })
        svg_data = tk.renderToSVG(1) 
        return svg_data
    except Exception as e:
        st.error(f"Verovio Render Error: {e}")
        return None

# --- 4. Main App Layout ---

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Settings")
    uploaded_file = st.file_uploader("Choose a song", type=["mp3", "wav"])
    
    selected_instrument_name = st.selectbox(
        "Select Instrument",
        options=list(INSTRUMENT_SETTINGS.keys())
    )
    
    # Renamed for clarity
    simplify_rhythm = st.checkbox("Simplify Rhythms (Snap to Eighth Notes)", value=True)
    
    inst_settings = INSTRUMENT_SETTINGS[selected_instrument_name]
    semitones = inst_settings["semitones"]
    is_monophonic = inst_settings["single_note"]

    if is_monophonic:
        st.caption(f"ℹ️ Mode: **Melody Only** (Overlaps/Chords removed)")
    else:
        st.caption(f"ℹ️ Mode: **Polyphonic**")

    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.button("Transcribe Audio"):
            with st.spinner("AI is listening..."):
                temp_audio_path = save_uploaded_file(uploaded_file)
                
                if temp_audio_path:
                    midi_data = audio_to_midi(temp_audio_path)
                    
                    if midi_data:
                        temp_xml_path = temp_audio_path.replace(os.path.splitext(temp_audio_path)[1], ".musicxml")
                        
                        success = midi_to_musicxml(
                            midi_data, 
                            temp_xml_path, 
                            semitones, 
                            is_monophonic,
                            simplify_rhythm
                        )
                        
                        if success:
                            st.session_state['xml_path'] = temp_xml_path
                            st.session_state['svg_data'] = render_svg(temp_xml_path)
                            st.success("Generation Complete!")
                    
                    os.remove(temp_audio_path)

with col2:
    st.header("2. Sheet Music Preview")
    
    if 'svg_data' in st.session_state and st.session_state['svg_data']:
        st.markdown(st.session_state['svg_data'], unsafe_allow_html=True)
        
        with open(st.session_state['xml_path'], "rb") as f:
            st.download_button(
                label="Download MusicXML",
                data=f,
                file_name="transcription.musicxml",
                mime="application/vnd.recordare.musicxml+xml"
            )
    else:
        st.info("Select an instrument and upload a file to begin.")