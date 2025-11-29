# Quickstart Guide: Thai Isan Lute (Phin) Music Transcription

## Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (optional, for version control)
- System with Python development headers
- Internet connection for downloading dependencies and audio data

## Setup Instructions

### 1. Clone the Repository (if applicable)
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install librosa pretty_midi yt-dlp torch mir_eval numpy scipy
```

### 4. Verify GPU Support (optional but recommended)
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Usage Examples

### 1. Download and Process Audio
```python
from src.data_pipeline.download import download_youtube_audio
from src.data_pipeline.feature_extraction import extract_phin_features

# Download audio from YouTube
audio_path = download_youtube_audio("https://www.youtube.com/watch?v=example", "./audio_sources")

# Extract CQT features
cqt_features = extract_phin_features(audio_path, sr=22050)
print(f"Extracted CQT features with shape: {cqt_features.shape}")
```

### 2. Run the Transcription Model
```python
from src.models.phin_transcriber import PhinTranscriber
from src.data_pipeline.feature_extraction import extract_phin_features

# Load your model
model = PhinTranscriber()

# Extract features from audio
cqt_features = extract_phin_features("./audio_sources/example.wav")

# Run transcription
transcription = model.transcribe(cqt_features)

# Save as MIDI
transcription.save_as_midi("./output/transcription.mid")
```

### 3. Evaluate Transcription Quality
```python
from src.evaluation.transcription_eval import evaluate_transcription

# Evaluate transcription against reference
onset_f1, pitch_f1 = evaluate_transcription(
    reference_midi_path="./references/example.mid",
    predicted_midi_path="./output/transcription.mid"
)

print(f"Onset F1 Score: {onset_f1:.3f}")
print(f"Pitch F1 Score: {pitch_f1:.3f}")
```

## Directory Structure
After setup, your project should have the following structure:
```
project-root/
├── src/
│   ├── setup/
│   ├── data_pipeline/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── audio_sources/          # Downloaded audio files
├── sheet_music/            # Reference sheet music (if available)
├── tests/                  # Test files
└── output/                 # Transcription results
```

## Troubleshooting

### Common Issues

1. **Audio processing errors**: Ensure ffmpeg is installed on your system
   ```bash
   # Install ffmpeg via conda
   conda install ffmpeg
   
   # Or via apt (Ubuntu/Debian)
   sudo apt update && sudo apt install ffmpeg
   ```

2. **PyTorch GPU errors**: If you have CUDA issues, install CPU-only PyTorch:
   ```bash
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory issues with large audio files**: Process audio in chunks or increase system virtual memory.

## Next Steps
1. Run the end-to-end pipeline: `python -m src.main`
2. Customize the model architecture in `src/models/phin_transcriber.py`
3. Add your own YouTube URLs to the batch download script
4. Experiment with different CQT parameters in feature extraction
5. Evaluate model performance against reference transcriptions