# Feature Specification: Thai Isan Lute (Phin) Music Transcription

**Feature Branch**: `001-phin-isan-transcription`
**Created**: 2025-11-29
**Status**: Draft
**Input**: User description: "นี่คือ Prompt ที่ละเอียดและอ้างอิงถึงเอกสารที่คุณสร้างไว้ ซึ่งเหมาะสมสำหรับใช้งานกับ AI Coding Agent เช่น Replit Agents, GitHub Copilot, หรือเครื่องมือ AI อื่นๆ ที่ใช้ในการสร้างโค้ด: 🤖 Prompt สำหรับ Agent: โครงการ AI Transcription พิณอีสาน (Phin Isan AI Dataset) ภารกิจ (Goal): สร้างโค้ด Python ที่ครอบคลุมขั้นตอนเริ่มต้นของโปรเจกต์ AI Transcription ลายพิณอีสาน โดยเน้นที่การติดตั้ง Dependencies, การจัดการ Data Pipeline (Download & Preprocessing), และการวางโครงสร้างพื้นฐานของโมเดล (Model Stub) เพื่อให้พร้อมสำหรับการเทรน แหล่งข้อมูลหลัก (References): อ้างอิงข้อมูลโค้ด, Dependencies, และขั้นตอนจากไฟล์ต่อไปนี้: quick_start_guide.md (Dependencies & Setup) youtube_sources.md (Download Script & URLs) training_pipeline.md (Feature Extraction & Evaluation) ขั้นตอนที่ Agent ต้องดำเนินการ (Actionable Steps): 1. Setup และติดตั้ง Dependencies (ไฟล์ setup.py) สร้าง Environment และติดตั้ง: สร้างโค้ดที่รันคำสั่ง pip install เพื่อติดตั้งไลบรารีทั้งหมดที่ระบุใน ขั้นตอนที่ 1 ของ quick_start_guide.md (ต้องมี librosa, pretty_midi, yt-dlp, torch หรือ tensorflow, mir_eval, ฯลฯ) ตรวจสอบ GPU และสร้างโฟลเดอร์: เพิ่มโค้ด Python เพื่อตรวจสอบว่า GPU (CUDA) พร้อมใช้งานหรือไม่ และสร้างโฟลเดอร์ที่จำเป็นตามโครงสร้างใน phin_dataset_master_guide.md (audio_sources/, sheet_music/, ฯลฯ) 2. Data Pipeline: ดาวน์โหลดและเตรียมเสียง (ไฟล์ data_pipeline.py) Batch Download Script: นำโค้ด Python สำหรับการดาวน์โหลดวิดีโอจาก "Batch Download Script" ใน youtube_sources.md มาปรับใช้ เป้าหมาย: ดาวน์โหลดวิดีโออย่างน้อย 3 ลิงก์แรก และแปลงเป็นไฟล์ .wav บันทึกในโฟลเดอร์ audio_sources/ Feature Extraction (CQT): สร้างฟังก์ชัน Python ชื่อ extract_phin_features(audio_path, sr=22050) ฟังก์ชันการทำงาน: โหลดไฟล์เสียง (librosa.load) ทำ Normalization ทำการดึงคุณลักษณะด้วย Constant-Q Transform (CQT) ตามที่แนะนำใน training_pipeline.md (เน้นว่า CQT เหมาะสำหรับดนตรีที่ไม่ใช่ 12-TET) ฟังก์ชันต้องคืนค่า CQT Spectrogram (NumPy Array) ที่พร้อมเข้าสู่โมเดล 3. Model Stubs และ Evaluation (ไฟล์ model_core.py) Model Architecture Stub (PyTorch/TensorFlow): สร้าง Class/Stub สำหรับโมเดลชื่อ PhinTranscriber โดยอิงตามสถาปัตยกรรม CNN-RNN-Attention ที่ระบุใน training_pipeline.md (ไม่ต้องใส่รายละเอียด input_size ที่ซับซ้อน แต่ให้ระบุเลเยอร์หลัก เช่น nn.Conv2d, nn.LSTM/nn.GRU, nn.Linear) Evaluation Function: คัดลอกและวางฟังก์ชัน evaluate_transcription(...) ที่ใช้ไลบรารี mir_eval.transcription จากส่วน "Evaluation Metrics" ของ training_pipeline.md เพื่อใช้ในการวัดค่า Onset F1 และ Pitch F1 (ต้องมีการอิมพอร์ต mir_eval และ pretty_midi) รูปแบบผลลัพธ์ที่ต้องการ: สร้างไฟล์ Python 3 ไฟล์ (หรือรวมในไฟล์เดียว) ที่มีโค้ดครบถ้วนตามขั้นตอนข้างต้น พร้อมคำอธิบายโค้ดโดยย่อ: setup.py data_pipeline.py model_core.py (รวมฟังก์ชัน Evaluation)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Environment Setup and Dependencies Installation (Priority: P1)

Music researchers and AI developers want to set up a complete Python environment with all necessary dependencies for Thai Isan lute music transcription, including audio processing libraries, machine learning frameworks, and YouTube download tools.

**Why this priority**: This is the foundational requirement - without a properly configured environment, no other functionality can be implemented or tested. This enables all subsequent development work.

**Independent Test**: Can be fully tested by running the setup script and verifying that all required libraries (librosa, pretty_midi, torch/tensorflow, mir_eval, yt-dlp) are successfully installed and accessible.

**Acceptance Scenarios**:

1. **Given** a fresh Python environment, **When** the setup script is executed, **Then** all required dependencies are installed successfully
2. **Given** a system with GPU support, **When** environment setup is performed, **Then** the system detects and configures GPU support for faster processing

---

### User Story 2 - Thai Isan Music Data Collection and Preprocessing (Priority: P2)

Cultural preservationists and machine learning engineers want to automatically download authentic Thai Isan music videos from YouTube and convert them to audio files suitable for transcription model training, following the traditional 7-tone musical system.

**Why this priority**: This is essential for creating the training dataset - without quality Isan music samples, the transcription model cannot learn the unique characteristics of Phin lute music.

**Independent Test**: Can be tested by running the batch download script with YouTube URLs and verifying that the audio files are downloaded, converted to .wav format, and stored in the correct directory structure.

**Acceptance Scenarios**:

1. **Given** a list of YouTube URLs containing Isan music, **When** the batch download script executes, **Then** .wav files are saved to the audio_sources/ directory
2. **Given** downloaded audio files, **When** feature extraction runs, **Then** CQT spectrograms are generated that represent the unique characteristics of Thai Isan music

---

### User Story 3 - Music Transcription Model Framework (Priority: P3)

AI researchers and musicologists want a basic model architecture that can transcribe Thai Isan lute music using Constant-Q Transform features and appropriate evaluation metrics to measure transcription accuracy.

**Why this priority**: This provides the core functionality framework for music transcription, enabling evaluation of onset and pitch accuracy specific to Thai musical traditions.

**Independent Test**: Can be tested by running the evaluation function on sample transcriptions and verifying that Onset F1 and Pitch F1 metrics are calculated correctly.

**Acceptance Scenarios**:

1. **Given** audio input with Isan music, **When** PhinTranscriber processes the features, **Then** it produces MIDI-like output representing the played notes
2. **Given** transcription results, **When** evaluation function runs, **Then** it returns accurate Onset F1 and Pitch F1 scores

---

### Edge Cases

- What happens when the downloaded audio is corrupted or has very poor quality?
- How does the system handle different recording environments (studio vs. live performance)?
- What if the Constant-Q Transform fails on particularly complex polyphonic audio?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST install all required Python dependencies (librosa, pretty_midi, yt-dlp, torch/tensorflow, mir_eval, etc.)
- **FR-002**: System MUST detect and utilize available GPU resources for accelerated processing
- **FR-003**: System MUST create the appropriate directory structure (audio_sources/, sheet_music/, etc.) for organizing data
- **FR-004**: System MUST download YouTube videos and convert them to .wav format
- **FR-005**: System MUST extract Constant-Q Transform (CQT) features from audio files suitable for Thai Isan music transcription
- **FR-006**: System MUST provide a model architecture stub (PhinTranscriber) with CNN-RNN-Attention layers
- **FR-007**: System MUST implement evaluation functions to measure Onset F1 and Pitch F1 accuracy
- **FR-008**: The extract_phin_features function MUST return a normalized CQT spectrogram compatible with the model input

### Key Entities

- **Audio Source**: Raw audio files (in .wav format) extracted from YouTube videos containing Thai Isan lute music
- **CQT Feature**: Constant-Q Transform spectrogram representation of audio, optimized for non-12-TET musical systems like Thai traditional music
- **Transcription Output**: MIDI-like representation of detected musical notes with timing and pitch information
- **Model Architecture**: CNN-RNN-Attention framework designed for music transcription tasks with Thai Isan music characteristics

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of required dependencies are installed successfully in a fresh Python environment
- **SC-002**: At least 3 YouTube videos are successfully downloaded and converted to .wav format within 10 minutes
- **SC-003**: CQT feature extraction processes a 30-second audio clip in under 30 seconds with appropriate output dimensions
- **SC-004**: The PhinTranscriber model stub successfully accepts CQT spectrogram input and produces structured output
- **SC-005**: Evaluation functions return meaningful Onset F1 and Pitch F1 scores when comparing generated transcriptions to reference MIDI files