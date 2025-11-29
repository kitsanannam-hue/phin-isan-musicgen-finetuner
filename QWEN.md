# Thai Isan Lute (Phin) Music Transcription Project

## Project Overview

This is a specialized Python-based AI music transcription system for Thai Isan lute (Phin) music. The project converts traditional Thai Isan music audio recordings into MIDI-like transcriptions with a focus on preserving the unique characteristics of the Thai 7-tone scale system and Phin lute playing techniques. This system serves both as a research tool and a cultural preservation effort for Thai traditional music.

## Key Technologies & Dependencies

- **Python 3.9+** (required for compatibility with deep learning libraries)
- **PyTorch** - For deep learning model implementation
- **librosa** - Audio processing and feature extraction
- **pretty_midi** - MIDI file handling
- **yt-dlp** - YouTube audio download functionality
- **mir_eval** - Music transcription evaluation metrics
- **numpy, scipy, matplotlib** - Scientific computing and visualization
- **Google APIs** - For additional data sources

## Architecture

### Core Components

1. **Data Pipeline** (`src/data_pipeline/`)
   - `download.py` - YouTube audio download functionality
   - `feature_extraction.py` - CQT feature extraction optimized for Thai music
   - `thai_isan_analysis.py` - Thai Isan music analysis
   - `training_data_preparer.py` - Training data preparation

2. **Model Architecture** (`src/models/`)
   - `phin_transcriber.py` - CNN-RNN-Attention model for music transcription

3. **Setup & Environment** (`src/setup/`)
   - `environment.py` - Environment setup and dependency installation

4. **Evaluation Framework** (`src/evaluation/`)
   - `transcription_eval.py` - Onset F1 and Pitch F1 evaluation functions

5. **Utilities** (`src/utils/`)
   - `constants.py` - Thai music system specific constants (7-tone scale, Phin parameters)

6. **Main System Integration**
   - `transcription_system.py` - Complete transcription system integration
   - `create_training_data.py` - Training data creation example

### Thai 7-tone Scale System

Thai traditional music uses a heptatonic scale system with specific frequency ratios:
- Tonic (Do): 1.0 ratio
- Second (Re): 1.125 ratio (9/8)
- Third (Mi): 1.25 ratio (5/4)
- Fourth (Fa): 1.333 ratio (4/3 approx)
- Fifth (So): 1.5 ratio (3/2)
- Sixth (La): 1.667 ratio (5/3 approx)
- Seventh (Ti): 1.789 ratio (augmented sixth approx)

### Phin Lute Characteristics

The system is specifically designed to handle the unique characteristics of the Phin lute:
- 3-string traditional instrument with common tuning: D3, A3, D4
- Unique timbral qualities
- Specific playing techniques (plucking, sliding, vibrato)
- Characteristic melodic patterns in Thai Isan music

## Building and Running

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify GPU support (optional but recommended):
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   ```

### Usage Examples

#### 1. Environment Setup
```python
from src.setup.environment import setup_environment

setup_info = setup_environment()
print(f"Dependencies installed: {setup_info['dependencies_installed']}")
print(f"GPU available: {setup_info['gpu_available']}")
```

#### 2. Audio Processing and Feature Extraction
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

#### 3. Music Transcription
```python
from src.transcription_system import ThaiIsanTranscriptionSystem

# Initialize the system
system = ThaiIsanTranscriptionSystem()

# Transcribe audio
transcription_result = system.transcribe_audio("path/to/thai_isan_audio.wav")
print(f"Transcribed {len(transcription_result['note_events'])} notes")
```

#### 4. Training Data Preparation
```python
from src.data_pipeline.training_data_preparer import ThaiIsanTrainingDataPreparer

# Prepare training data
preparer = ThaiIsanTrainingDataPreparer("./training_data")
audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # Your audio files
splits = preparer.prepare_training_data(audio_paths)
print(f"Created {len(splits['train'])} training samples")
```

## Evaluation Metrics

The system uses standard music transcription evaluation metrics:
- **Onset F1**: Measures accuracy of note onset detection
- **Pitch F1**: Measures accuracy of pitch classification
- **Thai Scale Adherence**: Measures how well transcriptions follow the Thai 7-tone scale

## Cultural Preservation Goals

This project prioritizes the preservation of Thai Isan musical heritage by:
- Supporting the traditional Thai 7-tone scale system
- Maintaining Phin lute playing patterns
- Capturing characteristic rhythmic structures
- Ensuring accurate note timing and pitch relationships

## Project Structure

```
project-root/
├── specs/                          # Project specifications and plans
│   └── 001-phin-isan-transcription/
│       ├── plan.md                 # Implementation plan
│       ├── spec.md                 # Detailed specification
│       ├── data-model.md           # Data model definition
│       └── ...
├── src/                            # Source code
│   ├── setup/
│   │   └── environment.py
│   ├── data_pipeline/
│   │   ├── download.py
│   │   ├── feature_extraction.py
│   │   ├── thai_isan_analysis.py
│   │   └── training_data_preparer.py
│   ├── models/
│   │   └── phin_transcriber.py
│   ├── evaluation/
│   │   └── transcription_eval.py
│   ├── utils/
│   │   └── constants.py
│   ├── transcription_system.py     # Main integration
│   ├── create_training_data.py
│   └── main.py                     # Entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_SUMMARY.md              # Project summary
└── thai_isan_analysis_demo.py      # Analysis demonstration
```

## Development Conventions

The project follows Python best practices with a focus on:
- Cultural accuracy and preservation
- Scientific validation of transcription quality
- Reproducible research with version control
- Proper evaluation metrics (Onset F1 and Pitch F1)
- Modular design with clear separation of concerns