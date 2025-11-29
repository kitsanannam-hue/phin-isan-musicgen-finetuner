# Implementation Plan: Thai Isan Lute (Phin) Music Transcription

**Branch**: `001-phin-isan-transcription` | **Date**: 2025-11-29 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-phin-isan-transcription/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This project implements an AI transcription system for Thai Isan lute (Phin) music that can convert audio recordings of traditional Thai Isan music into MIDI-like transcriptions. The system includes environment setup with necessary dependencies (librosa, pretty_midi, yt-dlp, PyTorch), data pipeline for downloading and preprocessing Isan music from YouTube, feature extraction using Constant-Q Transform (CQT) optimized for Thai 7-tone musical system, and a model architecture stub based on CNN-RNN-Attention for music transcription with appropriate evaluation metrics.

## Technical Context

**Language/Version**: Python 3.9+ (required for compatibility with AudioCraft and deep learning libraries)
**Primary Dependencies**: librosa (audio processing), pretty_midi (MIDI handling), yt-dlp (YouTube downloads), torch (PyTorch for ML), mir_eval (evaluation), numpy, scipy
**Storage**: Local file system for audio data and model artifacts
**Testing**: pytest with custom audio validation tests
**Target Platform**: Linux server/development environment (GPU-enabled preferred)
**Project Type**: Single project with audio processing and ML components
**Performance Goals**: CQT feature extraction under 30 seconds per 30-second audio clip; model inference under 5 seconds per clip
**Constraints**: Must support Thai 7-tone musical system (non-12-TET), GPU acceleration for model training, memory efficient processing of audio files
**Scale/Scope**: Single user research tool for cultural preservation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Post-design evaluation: The project fully aligns with the AudioCraft constitution:
- Research-First: This is fundamentally a research project for music transcription with scientific validation
- High-Quality Audio Generation: CQT feature extraction and CNN-RNN-Attention model preserve audio quality characteristics
- Reproducible Research: All experiments will be reproducible with fixed seeds, documented processes, and version control
- Model Training & Evaluation: Includes proper evaluation metrics (Onset F1 and Pitch F1) for rigorous testing
- Model Deployment & Distribution: Follows standardized model card documentation and appropriate licensing

## Project Structure

### Documentation (this feature)

```text
specs/001-phin-isan-transcription/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── setup/
│   ├── __init__.py
│   └── environment.py      # Environment setup and dependency installation
├── data_pipeline/
│   ├── __init__.py
│   ├── download.py         # YouTube audio download functionality
│   └── feature_extraction.py # CQT feature extraction for Thai Isan music
├── models/
│   ├── __init__.py
│   └── phin_transcriber.py # PhinTranscriber model architecture (CNN-RNN-Attention)
├── evaluation/
│   ├── __init__.py
│   └── transcription_eval.py # Onset F1 and Pitch F1 evaluation functions
└── utils/
    ├── __init__.py
    └── constants.py          # Thai music system specific constants (7-tone scale)
```

tests/
├── unit/
│   ├── test_feature_extraction.py
│   ├── test_model_architecture.py
│   └── test_evaluation.py
├── integration/
│   └── test_end_to_end.py
└── data/
    └── test_audio_samples/

**Structure Decision**: Single project structure selected to house all components of the music transcription system in a cohesive manner, with clear separation between setup, data pipeline, models, evaluation, and utility functions.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
