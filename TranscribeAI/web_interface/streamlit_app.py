"""
Streamlit web interface for TranscribeAI.
"""

import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transcription_model import TranscriptionPipeline


# Page configuration
st.set_page_config(
    page_title="TranscribeAI - Music Transcription",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def plot_waveform(audio, sr):
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig


def plot_spectrogram(spectrogram, sr, hop_length):
    """Plot spectrogram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap='viridis'
    )
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_pitch_contour(times, frequencies, voiced_flags, onset_times=None):
    """Plot pitch contour with onsets."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot pitch contour
    valid_mask = voiced_flags & (frequencies > 0) & np.isfinite(frequencies)
    ax.plot(times[valid_mask], frequencies[valid_mask], 
            linewidth=2, label='Detected Pitch', color='#1E88E5')
    
    # Plot onset markers
    if onset_times is not None and len(onset_times) > 0:
        for onset in onset_times:
            ax.axvline(x=onset, color='red', linestyle='--', 
                      alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Pitch Contour with Note Onsets')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 TranscribeAI</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">'\
                'Automatic Music Transcription - Convert Audio to Sheet Music'\
                '</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Transcription mode
    transcription_mode = st.sidebar.selectbox(
        "Transcription Mode",
        ["Pitch Contour", "Onset-Based"],
        help="Pitch Contour: Continuous pitch tracking\nOnset-Based: Note segmentation"
    )
    
    # Additional settings
    quantize_timing = st.sidebar.checkbox(
        "Quantize Timing",
        value=True,
        help="Snap note timings to rhythmic grid"
    )
    
    show_visualizations = st.sidebar.checkbox(
        "Show Visualizations",
        value=True,
        help="Display audio analysis visualizations"
    )
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.markdown('<h2 class="subheader">📁 Upload Audio File</h2>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, OGG, FLAC)",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Upload a monophonic audio file (single instrument)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Transcribe button
        st.markdown("---")
        if st.button("🎼 Transcribe Audio", type="primary", use_container_width=True):
            with st.spinner("Transcribing audio... This may take a moment."):
                try:
                    # Initialize pipeline
                    config_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        'config.yaml'
                    )
                    pipeline = TranscriptionPipeline(config_path)
                    
                    # Transcribe
                    if transcription_mode == "Pitch Contour":
                        midi_path, musicxml_path = pipeline.transcribe(temp_path)
                    else:
                        midi_path, musicxml_path = pipeline.transcribe_with_onsets(
                            temp_path,
                            quantize=quantize_timing
                        )
                    
                    st.success("✅ Transcription completed successfully!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h2 class="subheader">📊 Results</h2>', 
                               unsafe_allow_html=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if midi_path and os.path.exists(midi_path):
                            with open(midi_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download MIDI",
                                    data=f,
                                    file_name=os.path.basename(midi_path),
                                    mime="audio/midi"
                                )
                    
                    with col2:
                        if musicxml_path and os.path.exists(musicxml_path):
                            with open(musicxml_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download MusicXML",
                                    data=f,
                                    file_name=os.path.basename(musicxml_path),
                                    mime="application/xml"
                                )
                    
                    # Analysis
                    st.markdown("---")
                    st.markdown('<h2 class="subheader">📈 Analysis</h2>', 
                               unsafe_allow_html=True)
                    
                    analysis = pipeline.analyze_audio(temp_path)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", f"{analysis['duration']:.2f}s")
                    
                    with col2:
                        st.metric("Sample Rate", f"{analysis['sample_rate']} Hz")
                    
                    with col3:
                        st.metric("Notes Detected", analysis['num_onsets'])
                    
                    with col4:
                        tempo = analysis.get('estimated_tempo', 0)
                        st.metric("Estimated Tempo", f"{tempo:.0f} BPM")
                    
                    # Pitch range
                    if analysis['pitch_range']['min_hz'] > 0:
                        st.markdown("**Pitch Range:**")
                        st.write(f"- Frequency: {analysis['pitch_range']['min_hz']:.1f} Hz "
                                f"to {analysis['pitch_range']['max_hz']:.1f} Hz")
                        st.write(f"- MIDI Notes: {analysis['pitch_range']['min_midi']:.0f} "
                                f"to {analysis['pitch_range']['max_midi']:.0f}")
                    
                    # Visualizations
                    if show_visualizations:
                        st.markdown("---")
                        st.markdown('<h2 class="subheader">🎨 Visualizations</h2>', 
                                   unsafe_allow_html=True)
                        
                        viz_data = pipeline.get_visualization_data(temp_path)
                        
                        # Waveform
                        st.pyplot(plot_waveform(
                            viz_data['audio'], 
                            viz_data['sample_rate']
                        ))
                        
                        # Spectrogram
                        st.pyplot(plot_spectrogram(
                            viz_data['mel_spectrogram'],
                            viz_data['sample_rate'],
                            512
                        ))
                        
                        # Pitch contour
                        st.pyplot(plot_pitch_contour(
                            viz_data['times'],
                            viz_data['frequencies'],
                            viz_data['voiced_flags'],
                            viz_data['onset_times']
                        ))
                
                except Exception as e:
                    st.error(f"❌ Error during transcription: {str(e)}")
                    st.exception(e)
    
    else:
        # Instructions when no file uploaded
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 📝 Instructions
        
        1. **Upload** a monophonic audio file (single instrument, non-percussive)
        2. **Configure** settings in the sidebar (optional)
        3. **Click** the "Transcribe Audio" button
        4. **Download** the generated MIDI and MusicXML files
        5. **View** analysis and visualizations
        
        ### 🎯 Best Results
        
        - Use clear, monophonic recordings (one note at a time)
        - Avoid background noise and reverb
        - Supported instruments: flute, violin, voice, trumpet, etc.
        - Audio formats: WAV, MP3, OGG, FLAC
        
        ### 📊 Features
        
        - **Pitch Detection**: Advanced pYIN algorithm for accurate pitch tracking
        - **Onset Detection**: Identifies note boundaries automatically
        - **MIDI Generation**: Creates standard MIDI files
        - **Sheet Music**: Generates printable MusicXML format
        - **Visualizations**: Audio waveform, spectrogram, and pitch contour
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>TranscribeAI - Automatic Music Transcription System</p>
        <p>Powered by Librosa, TensorFlow, PrettyMIDI, and Music21</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
