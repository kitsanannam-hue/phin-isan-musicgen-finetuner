# Research: Thai Isan Lute (Phin) Music Transcription

## Decision: Python Environment and Dependencies
**Rationale**: Selected Python 3.9+ with specific libraries (librosa, pretty_midi, yt-dlp, torch, mir_eval) for audio processing and machine learning. These libraries are industry standard for music information retrieval and provide the necessary tools for Constant-Q Transform and model implementation.

**Alternatives considered**: 
- TensorFlow vs PyTorch: PyTorch chosen for better research flexibility and AudioCraft compatibility
- Different audio processing libraries: librosa is the de facto standard for academic audio processing

## Decision: Constant-Q Transform (CQT) for Feature Extraction
**Rationale**: CQT is specifically chosen over Short-Time Fourier Transform (STFT) as it provides better frequency resolution for music signals, especially suitable for non-12-TET systems like Thai 7-tone scale. This aligns with the requirement to handle Thai musical traditions properly.

**Alternatives considered**:
- STFT: More common but less suitable for non-Western musical systems
- Mel-spectrograms: Good for speech but less precise for pitch detection
- Raw waveforms: Requires more complex models, less interpretable

## Decision: CNN-RNN-Attention Architecture for Transcription
**Rationale**: This architecture is proven effective for music transcription tasks. CNN layers extract local features, RNN layers capture temporal dependencies in music, and attention mechanisms focus on relevant features for transcription accuracy.

**Alternatives considered**:
- Pure CNN: Good for feature extraction but lacks temporal modeling
- Pure RNN: Captures temporal dependencies but may lose local features
- Transformer-only: SOTA for many tasks but computationally expensive for audio

## Decision: YouTube as Data Source
**Rationale**: YouTube contains vast amounts of authentic Thai Isan music performances. Using yt-dlp allows for reliable downloading and conversion to audio formats suitable for processing.

**Alternatives considered**:
- Commercial datasets: Often expensive and may not specifically target Isan music
- Self-recorded data: Time-consuming and may not represent the diversity of traditional performances
- Academic datasets: Limited availability for Thai Isan music specifically

## Decision: Onset F1 and Pitch F1 as Evaluation Metrics
**Rationale**: These metrics are standard in music transcription research and appropriately measure the accuracy of note onset detection and pitch classification, which are the core requirements for this transcription system.

**Alternatives considered**:
- Frame-level accuracy: Measures every time frame but less meaningful for music transcription
- Symbolic accuracy: Requires conversion to symbolic representation which may not be suitable for traditional Thai music
- Perceptual metrics: More aligned with human perception but harder to implement and evaluate

## Decision: 7-tone Thai Musical System Implementation
**Rationale**: Thai traditional music uses a 7-tone scale system (heptatonic) which differs from the 12-tone equal temperament (12-TET) of Western music. The implementation will account for these frequency relationships to properly handle Isan music's unique characteristics.

**Alternatives considered**:
- Adapting Western music tools directly: Would not properly represent Thai musical traditions
- Approximating with Western scales: Would lose important cultural musical characteristics