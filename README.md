# Thai Isan Lute (Phin) Music Transcription

This project implements an AI transcription system for Thai Isan lute (Phin) music that can convert audio recordings of traditional Thai Isan music into MIDI-like transcriptions. The system is specifically designed to handle the unique characteristics of the Thai 7-tone musical scale system and Phin lute playing techniques.

## Features

- **Thai 7-tone Scale Support**: Optimized for the traditional Thai heptatonic scale system
- **Phin Lute Specialization**: Features extraction and modeling tailored to the Phin lute characteristics
- **Accurate Note Capture**: High-precision transcription with focus on capturing every musical note
- **Cultural Preservation**: Maintains the unique musical characteristics of Thai Isan music
- **Constant-Q Transform**: Feature extraction optimized for non-Western musical systems
- **Comprehensive Evaluation**: Onset F1 and Pitch F1 metrics for transcription quality assessment

## Project Structure

```
project-root/
├── src/
│   ├── setup/
│   │   └── environment.py      # Environment setup and dependency installation
│   ├── data_pipeline/
│   │   ├── download.py         # YouTube audio download functionality
│   │   ├── feature_extraction.py # CQT feature extraction for Thai Isan music
│   │   ├── thai_isan_analysis.py # Thai Isan music analysis
│   │   ├── training_data_preparer.py # Training data preparation
│   │   └── __init__.py
│   ├── models/
│   │   └── phin_transcriber.py # PhinTranscriber model architecture (CNN-RNN-Attention)
│   ├── evaluation/
│   │   └── transcription_eval.py # Onset F1 and Pitch F1 evaluation functions
│   ├── utils/
│   │   └── constants.py        # Thai music system specific constants (7-tone scale)
│   ├── transcription_system.py # Complete transcription system integration
│   ├── create_training_data.py # Training data creation example
│   └── main.py                 # Main entry point
├── audio_sources/              # Downloaded audio files
├── sheet_music/                # Reference sheet music (if available)
├── output/                     # Transcription results
├── references/                 # Reference MIDI files
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd thai-isan-transcription
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify GPU support (optional but recommended):
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   ```

## Usage

### 1. Environment Setup
```python
from src.setup.environment import setup_environment

setup_info = setup_environment()
print(f"Dependencies installed: {setup_info['dependencies_installed']}")
print(f"GPU available: {setup_info['gpu_available']}")
```

### 2. Audio Processing and Feature Extraction
```python
from src.data_pipeline.feature_extraction import extract_phin_features
from src.data_pipeline.thai_isan_analysis import create_detailed_transcription_report

# Extract features optimized for Thai music
features = extract_phin_features("path/to/thai_isan_audio.wav")
print(f"Extracted features with shape: {features.shape}")

# Analyze cultural characteristics
analysis = create_detailed_transcription_report("path/to/thai_isan_audio.wav")
print(f"Thai scale adherence: {analysis['thai_scale_adherence']:.2%}")
```

### 3. Music Transcription
```python
from src.transcription_system import ThaiIsanTranscriptionSystem

# Initialize the system
system = ThaiIsanTranscriptionSystem()

# Transcribe audio
transcription_result = system.transcribe_audio("path/to/thai_isan_audio.wav")
print(f"Transcribed {len(transcription_result['note_events'])} notes")
```

### 4. Training Data Preparation
```python
from src.data_pipeline.training_data_preparer import ThaiIsanTrainingDataPreparer

# Prepare training data
preparer = ThaiIsanTrainingDataPreparer("./training_data")
audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # Your audio files
splits = preparer.prepare_training_data(audio_paths)
print(f"Created {len(splits['train'])} training samples")
```

## Thai 7-tone Scale System

Thai traditional music uses a 7-tone (heptatonic) scale system that differs from the 12-tone equal temperament (12-TET) of Western music. The implementation accounts for these frequency relationships:

- Tonic (Do): 1.0 ratio
- Second (Re): 1.125 ratio (9/8)
- Third (Mi): 1.25 ratio (5/4)
- Fourth (Fa): 1.333 ratio (4/3 approx)
- Fifth (So): 1.5 ratio (3/2)
- Sixth (La): 1.667 ratio (5/3 approx)
- Seventh (Ti): 1.789 ratio (augmented sixth approx)

## Phin Lute Characteristics

The system is specifically designed to handle the unique characteristics of the Phin lute:

- 3-string traditional instrument
- Common tuning: D3, A3, D4
- Unique timbral qualities
- Specific playing techniques (plucking, sliding, vibrato)
- Characteristic melodic patterns in Thai Isan music

## Evaluation Metrics

The system uses standard music transcription evaluation metrics:

- **Onset F1**: Measures accuracy of note onset detection
- **Pitch F1**: Measures accuracy of pitch classification
- **Thai Scale Adherence**: Measures how well transcriptions follow the Thai 7-tone scale

## Creating Training Data

The system includes tools for creating high-quality training data:

1. Download authentic Thai Isan music
2. Extract CQT features optimized for Thai music
3. Generate accurate note labels with Thai scale quantization
4. Validate cultural characteristics preservation
5. Split data into train/validation/test sets

## Contributing

Contributions to improve the Thai Isan music transcription system are welcome. Please ensure that any contributions respect and preserve the cultural authenticity of Thai traditional music.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thai musical traditions and the artists who preserve them
- The open-source audio processing community
- Researchers in music information retrieval