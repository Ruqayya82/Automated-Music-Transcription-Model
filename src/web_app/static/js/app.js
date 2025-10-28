// TranscribeAI Web App JavaScript

let uploadedFileName = null;
let midiFileName = null;
let xmlFileName = null;

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const uploadPrompt = uploadArea.querySelector('.upload-prompt');
const changeFileBtn = document.getElementById('change-file');
const optionsSection = document.getElementById('options-section');
const transcribeBtn = document.getElementById('transcribe-btn');
const useModelCheckbox = document.getElementById('use-model');
const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultsSection = document.getElementById('results-section');
const numNotesEl = document.getElementById('num-notes');
const methodEl = document.getElementById('method');
const downloadMidiBtn = document.getElementById('download-midi');
const downloadXmlBtn = document.getElementById('download-xml');
const newTranscriptionBtn = document.getElementById('new-transcription');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');

// Upload Area Click
uploadArea.addEventListener('click', () => {
    if (!uploadedFileName) {
        fileInput.click();
    }
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
});

// Change File Button
changeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Transcribe Button
transcribeBtn.addEventListener('click', () => {
    if (uploadedFileName) {
        transcribeAudio();
    }
});

// Download Buttons
downloadMidiBtn.addEventListener('click', () => {
    if (midiFileName) {
        downloadFile(midiFileName);
    }
});

downloadXmlBtn.addEventListener('click', () => {
    if (xmlFileName) {
        downloadFile(xmlFileName);
    }
});

// New Transcription Button
newTranscriptionBtn.addEventListener('click', () => {
    resetAll();
});

// Retry Button
retryBtn.addEventListener('click', () => {
    hideError();
    showOptions();
});

// Handle File Upload
async function handleFile(file) {
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/m4a', 'audio/x-m4a'];
    const allowedExtensions = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
        showError('Please upload a valid audio file (WAV, MP3, FLAC, OGG, M4A)');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    // Upload file
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        progressText.textContent = 'Uploading file...';
        showProgress();
        setProgress(30);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            uploadedFileName = data.filename;
            fileName.textContent = `✓ ${file.name}`;
            uploadPrompt.style.display = 'none';
            fileInfo.style.display = 'block';
            hideProgress();
            showOptions();
            setProgress(0);
        } else {
            showError(data.error || 'Upload failed');
        }
    } catch (error) {
        showError('Failed to upload file. Please try again.');
        console.error(error);
    }
}

// Transcribe Audio
async function transcribeAudio() {
    hideOptions();
    hideError();
    showProgress();
    setProgress(10);
    progressText.textContent = 'Initializing transcription...';
    
    const useModel = useModelCheckbox.checked;
    
    try {
        setProgress(30);
        progressText.textContent = 'Analyzing audio...';
        
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: uploadedFileName,
                use_model: useModel
            })
        });
        
        setProgress(60);
        progressText.textContent = 'Detecting pitch and onsets...';
        
        const data = await response.json();
        
        if (data.success) {
            setProgress(80);
            progressText.textContent = 'Generating MIDI and MusicXML...';
            
            // Store file names
            midiFileName = data.midi_file;
            xmlFileName = data.musicxml_file;
            
            setProgress(100);
            progressText.textContent = 'Complete!';
            
            setTimeout(() => {
                hideProgress();
                showResults(data);
            }, 500);
        } else {
            showError(data.error || 'Transcription failed');
        }
    } catch (error) {
        showError('Failed to transcribe audio. Please try again.');
        console.error(error);
    }
}

// Download File
function downloadFile(filename) {
    window.location.href = `/api/download/${filename}`;
}

// Show/Hide Sections
function showOptions() {
    optionsSection.style.display = 'block';
}

function hideOptions() {
    optionsSection.style.display = 'none';
}

function showProgress() {
    progressSection.style.display = 'block';
}

function hideProgress() {
    progressSection.style.display = 'none';
}

function showResults(data) {
    numNotesEl.textContent = data.num_notes;
    methodEl.textContent = data.metadata.method === 'model' ? 'ML Model' : 'Traditional';
    resultsSection.style.display = 'block';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    hideProgress();
}

function hideError() {
    errorSection.style.display = 'none';
}

function setProgress(percent) {
    progressFill.style.width = `${percent}%`;
}

// Reset Functions
function resetUpload() {
    uploadedFileName = null;
    uploadPrompt.style.display = 'block';
    fileInfo.style.display = 'none';
    fileInput.value = '';
    hideOptions();
}

function resetAll() {
    resetUpload();
    hideResults();
    hideError();
    hideProgress();
    midiFileName = null;
    xmlFileName = null;
    setProgress(0);
}

// Check API Status on Load
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('API Status:', data);
    } catch (error) {
        console.error('Failed to check API status:', error);
    }
}

// Initialize
checkStatus();
