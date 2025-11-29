# Data Model: Thai Isan Lute (Phin) Music Transcription

## Entities

### AudioSource
- **id**: String - Unique identifier for the audio source
- **title**: String - Title of the audio/video source (from YouTube)
- **url**: String - URL of the original source
- **file_path**: String - Local path to the .wav file after download
- **duration**: Float - Duration of the audio in seconds
- **sample_rate**: Integer - Sample rate of the audio file (default 22050Hz)
- **download_date**: DateTime - When the audio was downloaded
- **status**: Enum ['pending', 'downloaded', 'processed', 'failed'] - Processing status

### CQTFeature
- **id**: String - Unique identifier for the feature extraction
- **audio_source_id**: String - Reference to the AudioSource
- **cqt_matrix**: Array<Array<Float>> - 2D array representing the CQT spectrogram
- **frequencies**: Array<Float> - Frequency bins of the CQT
- **time_stamps**: Array<Float> - Time stamps for each column of the CQT matrix
- **n_bins**: Integer - Number of frequency bins in the CQT
- **hop_length**: Integer - Hop length used for CQT computation
- **creation_date**: DateTime - When the features were extracted

### TranscriptionOutput
- **id**: String - Unique identifier for the transcription
- **audio_source_id**: String - Reference to the AudioSource
- **cqt_feature_id**: String - Reference to the CQTFeature used
- **midi_data**: String - MIDI representation of the transcription
- **notes**: Array<Object> - Array of note objects with properties:
  - start_time: Float - Start time of the note
  - end_time: Float - End time of the note
  - pitch: Integer - MIDI pitch number
  - velocity: Integer - Note velocity/amplitude
- **confidence_score**: Float - Overall confidence of the transcription
- **creation_date**: DateTime - When the transcription was created

### PhinTranscriberModel
- **id**: String - Model identifier
- **name**: String - Name of the model ("PhinTranscriber")
- **architecture**: String - Model architecture ("CNN-RNN-Attention")
- **input_shape**: Array<Integer> - Expected input dimensions [time, frequencies]
- **output_shape**: Array<Integer> - Expected output dimensions
- **training_status**: Enum ['untrained', 'training', 'trained', 'evaluated']
- **created_date**: DateTime - When model architecture was defined
- **last_modified**: DateTime - When model was last modified

### EvaluationResult
- **id**: String - Unique identifier for the evaluation
- **transcription_id**: String - Reference to the TranscriptionOutput
- **onset_f1_score**: Float - Onset F1 score (0.0-1.0)
- **pitch_f1_score**: Float - Pitch F1 score (0.0-1.0)
- **onset_precision**: Float - Onset precision (0.0-1.0)
- **onset_recall**: Float - Onset recall (0.0-1.0)
- **pitch_precision**: Float - Pitch precision (0.0-1.0)
- **pitch_recall**: Float - Pitch recall (0.0-1.0)
- **evaluation_date**: DateTime - When the evaluation was performed

## Relationships
- AudioSource --(1 to many)--> CQTFeature
- CQTFeature --(1 to 1)--> TranscriptionOutput
- AudioSource --(1 to 1)--> TranscriptionOutput
- TranscriptionOutput --(1 to 1)--> EvaluationResult
- PhinTranscriberModel --(1 to many)--> TranscriptionOutput

## Validation Rules
- Audio files must be in .wav format after download
- CQT matrices must have consistent dimensions for batch processing
- Transcription notes must have valid MIDI pitch values (0-127)
- Evaluation scores must be between 0.0 and 1.0
- Time stamps must be within the bounds of the audio duration

## State Transitions
- AudioSource: pending → downloaded → processed (→ failed)
- PhinTranscriberModel: untrained → training → trained → evaluated