# 📚 ข้อมูลวิจัยสำหรับโปรเจค Isan Musical Instruments Classifier
## รายงานการสืบค้นข้อมูลเชิงลึก - วันที่ 24 พฤศจิกายน 2025

---

## 📋 สารบัญ

1. [ภาพรวมโปรเจค](#ภาพรวมโปรเจค)
2. [แหล่งข้อมูลโน๊ตลายพิณและดนตรีไทย](#แหล่งข้อมูลโน๊ตลายพิณและดนตรีไทย)
3. [Datasets สำหรับ Music Transcription](#datasets-สำหรับ-music-transcription)
4. [โปรเจค Open Source และโมเดล AI](#โปรเจค-open-source-และโมเดล-ai)
5. [งานวิจัยที่เกี่ยวข้อง](#งานวิจัยที่เกี่ยวข้อง)
6. [แหล่งเสียงและวิดีโอบน YouTube](#แหล่งเสียงและวิดีโอบน-youtube)
7. [แนวทางการพัฒนาโมเดล](#แนวทางการพัฒนาโมเดล)
8. [ข้อมูลเทคนิคจากโปรเจคปัจจุบัน](#ข้อมูลเทคนิคจากโปรเจคปัจจุบัน)

---

## 🎯 ภาพรวมโปรเจค

### วัตถุประสงค์
สร้างระบบ AI สำหรับจำแนกและวิเคราะห์เครื่องดนตรีพื้นบ้านอีสาน โดยเฉพาะ:
- **พิณ (Phin)** - พิณสามสายจากภาคตะวันออกเฉียงเหนือ
- **แคน (Khaen)** - แคนปี่ไม้ไผ่ เครื่องดนตรีประจำชาติลาว

### เทคโนโลยีหลัก
- Audio Processing: librosa, soundfile
- Machine Learning: scikit-learn (Random Forest)
- Feature Engineering: MFCC, Chroma, Spectral Analysis
- Web Interface: Streamlit
- Visualization: Plotly, Matplotlib, Seaborn

---

## 🎵 แหล่งข้อมูลโน๊ตลายพิณและดนตรีไทย

### 1. โน๊ตลายพิณพื้นฐาน
**แหล่ง:** guitar285.wordpress.com

#### ลายพิณยอดนิยม (Pentatonic Scale):

**ลายนกไส่บินข้ามทุ่ง**
```
* / ม ซ ล ล ด ซ ล / ม ซ ล ล ด ซ ล / 
ด ร ม ม ซ ร ม / ด ร ม ม ซ ร ม / 
ม ซ ล ล ด ซ ล / ม ซ ล ล ด ซ ล / 
ม ร ด ล ด ซ ล / ม ร ด ล ด ซ ล / 
(ซ้ำ * 5 รอบ)
```

**ลายแมลงภู่ตอมดอกไม้**
```
(เกริ่น) ด . . . . ./ ร . . . . ./ ม . . . . ./
* / ล ล ล / ล ม / ม ซ ม / ร ม ซ ม /
ล ล ล / ล ร / ด ร ม ร ด / ล ด ล ม /
(กลับไปซ้ำ *)
```

**ลายโปงลาง**
```
* / ม ซ ล ซ ล / ล ด ล ซ ล / 
ด ร ม ร ม / ล ซ ม ร ม / 
ม ซ ล ซ ล / ร ด ล ซ ล /
(ซ้ำ * 5 รอบ)
```

**ลายเซิ้งบั้งไฟ**
```
* / ล ด ล ด ล ด ล / ล ด ล ด ล ด ล / 
ล ด ล ด ร ด ล / ร ด ร ด ร ด ล /
ม ร ม ร ม ด ร / ม ร ม ร ม ด ร /
(ซ้ำ * 5 รอบ)
```

**ลายเต้ยโขง**
```
* / ล ซ ม ล / ซ ด ล ซ ล ม / 
ล ซ ม ล / ซ ด ม ล / 
ซ ด ล ซ ม ล / ซ ม ร ด ม /
(ซ้ำ * 5 รอบ)
```

#### ระบบโน๊ตดนตรีไทย
- **Scale:** ระบบ 7 เสียงที่กระจายเท่าๆ กัน (7-tone equidistant)
- **โซลฟาไทย:** ด ร ม ฟ ซ ล ท
- **แตกต่างจาก Western:** ดนตรีตะวันตกใช้ 12 เสียงต่อ octave
- **Pentatonic (5 เสียง):** พื้นฐานของเพลงไทยส่วนใหญ่

### 2. งานวิจัยเกี่ยวกับลายพิณ
**ชื่อ:** "การศึกษาลายพิณในวัฒนธรรมดนตรีอีสาน กรณีศึกษา: จังหวัดอุบลราชธานี"
**ผู้วิจัย:** ธงไท จันเต (2554)
**สถาบัน:** มหาวิทยาลัยบูรพา

**ข้อค้นพบสำคัญ:**
- พิณมีประวัติมาจากภาษาบาลี-สันสกฤต "วีณา"
- นำเข้ามาในไทยประมาณ 1,000 ปีที่แล้ว
- ใช้ในจังหวัดอุบลราชธานีประมาณ 200 ปี
- พัฒนาจากพิณสองสาย → สามสาย → สี่สาย
- **บันไดเสียง:** Pentatonic Scale เป็นหลัก
- **เทคนิค:** มีการบรรเลงแบบ improvisation (ด้นสด) คล้าย Jazz

**ลายพิณที่ศึกษา:**
1. ลายกาเต้นก้อน
2. ลายปู่ป๋าหลาน
3. ลายลำเพลิน
4. ลายมโหรีอีสาน

---

## 📊 Datasets สำหรับ Music Transcription

### 1. NSynth Dataset (Google Magenta)
- **ขนาด:** 305,979 musical notes
- **เครื่องดนตรี:** 1,006 instruments
- **คุณสมบัติ:** แต่ละโน๊ตมี unique pitch, timbre, envelope
- **ลิงก์:** https://magenta.withgoogle.com/datasets/nsynth
- **ข้อจำกัด:** มุ่งเน้นดนตรีตะวันตก (12-tone system)

### 2. OpenMIC-2018
- **ขนาด:** 20,000 audio clips (10 วินาทีต่อคลิป)
- **แหล่งที่มา:** Free Music Archive (Creative Commons)
- **Labels:** Multi-instrument recognition
- **จำนวน Instruments:** 8-10 classes
- **ดาวน์โหลด:** 
  - GitHub: https://github.com/cosmir/openmic-2018
  - Zenodo: https://zenodo.org/records/1432913
- **ข้อดี:** Open source, crowd-sourced labels

### 3. FMA (Free Music Archive)
- **ขนาดทั้งหมด:** 917 GiB
- **จำนวนเพลง:** 106,574 tracks
- **Metadata:** ชื่อเพลง, อัลบั้ม, ศิลปิน, แนว, tags, description
- **Subsets:**
  - **fma_small:** 8,000 tracks (30 วินาที, 8 genres) - 7.2 GiB
  - **fma_medium:** 25,000 tracks - 25 GiB  
  - **fma_large:** 106,574 tracks - 93 GiB
  - **fma_full:** พร้อม full-length audio - 917 GiB
- **ดาวน์โหลด:**
  - GitHub: https://github.com/mdeff/fma
  - Kaggle: https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium
  - Hugging Face: https://huggingface.co/datasets/benjamin-paine/free-music-archive-full
- **ประโยชน์:** เหมาะสำหรับ genre classification และ instrument recognition

### 4. MusicNet
- **ขนาด:** 330 recordings of classical music
- **คีย์:** 10 composers, 11 instruments
- **Labels:** Over 1 million annotated labels (frame-level)
- **ข้อดี:** frame-level timbre transcription
- **ลิงก์:** https://benadar293.github.io/

### 5. Slakh2100 Dataset
- **ขนาด:** 2,100 tracks
- **คุณสมบัติ:** Multi-track, synthesized music
- **งานที่เหมาะสม:**
  - Music Instrument Recognition
  - Automatic Music Transcription (AMT)
  - Music Source Separation (MSS)
- **GitHub:** https://github.com/KinWaiCheuk/slakh_loader

### 6. HamNava Dataset (Persian/Iranian Music)
- **ขนาด:** 6,000 audio excerpts
- **คุณสมบัติ:** Multi-label instrument classification
- **ระบบ:** Dastgah (modal system คล้ายดนตรีไทย)
- **ข้อดี:** ตัวอย่างดี ๆ สำหรับ non-Western music
- **ลิงก์:** https://transactions.ismir.net/articles/10.5334/tismir.257

### 7. GigaMIDI Dataset
- **ขนาด:** 1.4 ล้าน MIDI files
- **แหล่งที่มา:** Internet Archive
- **ประโยชน์:** symbolic music representation
- **ลิงก์:** https://transactions.ismir.net/articles/10.5334/tismir.203

---

## 🛠️ โปรเจค Open Source และโมเดล AI

### 1. Spotify Basic Pitch ⭐ (แนะนำ)
**GitHub:** https://github.com/spotify/basic-pitch

**คุณสมบัติ:**
- Lightweight neural network สำหรับ Audio-to-MIDI
- รองรับ pitch bend detection
- Instrument-agnostic (ทำงานกับเครื่องดนตรีหลากหลาย)
- รองรับ polyphonic instruments
- มี Web Demo: https://basicpitch.spotify.com/

**Model Formats:**
- TensorFlow (original)
- CoreML (MacOS)
- TensorFlowLite (Linux)
- ONNX (Windows)

**การติดตั้ง:**
```bash
pip install basic-pitch
```

**การใช้งาน:**
```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

model_output, midi_data, note_events = predict(<audio_path>)
```

**ข้อดี:**
- เล็กและเร็ว
- ใช้งานง่าย
- รองรับหลาย platform
- Pitch bend สำหรับเสียงที่ไม่ตรง Western scale

### 2. Omnizart
**GitHub:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
**Documentation:** https://music-and-culture-technology-lab.github.io/omnizart-doc/

**คุณสมบัติ:**
- Python library สำหรับ automatic music transcription
- รองรับหลายประเภท:
  - Pitched instruments
  - Vocal melody  
  - Chord transcription
  - Drum event detection
- มี pre-trained models

**การติดตั้ง:**
```bash
pip install omnizart
```

### 3. MT3 (Google Magenta)
**แนวคิด:** Multi-Task Multitrack Music Transcription
**GitHub:** https://github.com/google/flax/discussions/1664

**คุณสมบัติ:**
- Unified training framework
- Multi-instrument transcription
- Transformer-based architecture
- High-quality results

### 4. YourMT3+
**Paper:** https://arxiv.org/html/2407.04822v1

**คุณสมบัติ:**
- Enhanced multi-instrument transcription
- Language token decoding approach
- ปรับปรุงจาก MT3

### 5. NeuralNote (VST Plugin)
**GitHub:** https://github.com/DamRsn/NeuralNote

**คุณสมบัติ:**
- Audio plugin สำหรับ DAW
- รูปแบบ: VST3, Component, Standalone
- Real-time transcription
- ใช้ Basic Pitch ภายใต้ฝากระโปรง

### 6. งานวิจัย KMUTT (มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี)
**ชื่อ:** "Automatic Music Transcription for Thai Xylophone"
**ลิงก์:** https://inc.kmutt.ac.th/download/capstone_design_projects/2567/10.pdf

**วิธีการ:**
- Energy-based approach
- Spectral Analysis (EWMA, FFT, Peak Detection)
- **ไม่ใช่ deep learning**

**ผลลัพธ์:**
- **Onset detection F1-score:** 98.54%
- **Pitch detection F1-score:** 97.34%
- ดีกว่า deep learning models (Onsets and Frames)
- สามารถทำงานบน embedded devices ได้

**ข้อดี:**
- เหมาะสำหรับ real-time processing
- ใช้ทรัพยากรน้อย
- แม่นยำสูงสำหรับดนตรีไทย

---

## 📖 งานวิจัยที่เกี่ยวข้อง

### 1. ดนตรีไทยและ AI Bias
**ชื่อ:** "Stepping Towards Transcultural Machine Learning in Music" 
**ผู้จัดทำ:** Google Magenta
**ลิงก์:** https://magenta.withgoogle.com/transcultural

**ประเด็นสำคัญ:**
- AI models ส่วนใหญ่ออกแบบสำหรับดนตรีตะวันตก (12-tone)
- ดนตรีไทยใช้ 7-tone equidistant system
- ต้องการ dataset และ model ที่เข้าใจระบบดนตรีไม่ใช่ตะวันตก

**ข้อเสนอแนะ:**
- สร้าง training data จากดนตรีไทย
- Retrain หรือ fine-tune models
- พัฒนา evaluation metrics ที่เหมาะสม

### 2. Deep Learning สำหรับดนตรีไทย
**ชื่อ:** "Deep Learning for Music Genre Classification: Thai Music"
**ลิงก์:** https://dl.acm.org/doi/full/10.1145/3722150.3722157

**วิธีการ:**
- CNN และ RNN
- จำแนกประเภทเพลงไทย

### 3. Thai Tuning System Studies
**หัวข้อสำคัญ:**
- "The Myth of Equidistance in Thai Tuning"
- "Equiheptatonic Tuning in Thai Classical Music"
- "Relative Nature of Thai Traditional Music through its Tuning System"

**ข้อค้นพบ:**
- ระบบเสียงไทยเป็น 7-tone equidistant (ทฤษฎี)
- ในทางปฏิบัติมีความแปรผัน (variations)
- Pitch intervals ไม่เท่ากันทุกเครื่องดนตรี
- มีความยืดหยุ่นตามภูมิภาคและศิลปิน

---

## 🎥 แหล่งเสียงและวิดีโอบน YouTube

### 1. Tutorial Channels (สำหรับ Training Data)

**มูนมังอีสาน Channel**
- สอนพิณหลากหลายลาย
- มีโน๊ตประกอบ
- เหมาะสำหรับมือใหม่
- ลิงก์ตัวอย่าง: https://www.youtube.com/watch?v=-l1Pj7N_eI8

**ดุลย์เพลงพิณ**
- สอนพิณพื้นฐาน
- ลายพิณต่าง ๆ
- ไม่ต้องจำโน๊ต
- ลิงก์: https://www.youtube.com/watch?v=_Mm7gnrdy08

**ฐิติวัสส์ ทองอ่อน (สตีฟ)**
- สอนแบบละเอียด
- มีโน๊ตพิณหลายชุด
- รวมโน๊ตพิณ 7 ลาย, 8 ลาย, 23 ลาย
- Website: https://sites.google.com/view/stevethitiwat

**หนิง ซิงกิ รีวิวบรรเลง**
- ลายมโหรีอีสาน
- performance review
- ลิงก์: https://www.youtube.com/watch?v=daZpyFy1Qb8

**M MUSIC GROUP**
- การสอนแบบเป็นขั้นตอน
- ลายพิณพื้นฐาน
- มีโน๊ตและซาวด์ประกอบ

### 2. ลายพิณที่มีใน YouTube
1. ลายลำเพลิน
2. ลายแห่ (หลายแบบ)
3. ลายเลาะบ้าน
4. ลายมโหรีอีสาน
5. ลายโปงลาง
6. ลายเต้ย
7. ลายปู่ป๋าหลาน
8. ลายแมงตับเต่า

### 3. วิธีการรวบรวมข้อมูล

**เครื่องมือ:**
```bash
# ติดตั้ง yt-dlp
pip install yt-dlp

# ดาวน์โหลดเสียง
yt-dlp -x --audio-format wav <youtube_url>

# ดาวน์โหลด playlist
yt-dlp -x --audio-format wav <playlist_url>
```

**ตัวอย่างใน Project:**
```python
# examples/download_youtube_videos.py
from src.data_collection.youtube_downloader import YouTubeDownloader

downloader = YouTubeDownloader()
downloader.download_audio(url, output_path)
```

---

## 🔬 แนวทางการพัฒนาโมเดล

### Pipeline แนะนำ

#### 1. Data Collection Phase
```
1. รวบรวมวิดีโอจาก YouTube
   - ใช้ yt-dlp
   - แยกเสียงเป็น WAV (22,050 Hz)
   - จัดเก็บตามเครื่องดนตรี

2. บันทึก metadata
   - ชื่อเพลง/ลาย
   - เครื่องดนตรี  
   - ศิลปิน
   - consent status
   - cultural attribution

3. สร้าง ground truth
   - ใช้ Basic Pitch แปลงเป็น MIDI
   - ตรวจสอบความถูกต้อง
   - บันทึก annotations
```

#### 2. Data Preprocessing
```python
# ใช้ AudioProcessor ในโปรเจค
from src.preprocessing.audio_processor import AudioProcessor

processor = AudioProcessor(target_sr=22050)

# 1. โหลดและ normalize
audio, sr, quality = processor.preprocess_audio(file_path)

# 2. แบ่ง segments (3 วินาที, overlap 50%)
segments = processor.extract_segments(audio, sr, segment_duration=3.0)

# 3. ตรวจสอบคุณภาพ
if not quality['is_valid']:
    print(f"Audio issue: {quality.get('reason')}")
```

#### 3. Feature Extraction
```python
from src.features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(sr=22050)

# แตก features ทั้งหมด (143 features)
features = extractor.extract_all_features(audio)

# Features แยกตามประเภท:
# - MFCCs (20 coefficients × 4 statistics = 80 features)
# - Chroma (12 bins × 4 statistics = 48 features)
# - Spectral (centroid, rolloff, bandwidth)
# - Temporal (zero-crossing rate, RMS)
# - Pitch (fundamental frequency analysis)
```

#### 4. Model Training
```python
from src.models.classifier import InstrumentClassifier

# Random Forest Classifier
classifier = InstrumentClassifier(
    n_estimators=100,
    random_state=42
)

# Train
results = classifier.train(X, y, validation_split=0.2)

# Metrics
print(f"Accuracy: {results['train_accuracy']:.2%}")
print(f"Val Accuracy: {results['validation_accuracy']:.2%}")
print(f"CV Mean: {results['cv_mean']:.2%}")
```

#### 5. Evaluation
```python
from src.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Confusion Matrix
evaluator.plot_confusion_matrix(y_true, y_pred, labels)

# Feature Importance
importance = classifier.get_feature_importance(feature_names, top_n=20)
```

### แนวทางสำหรับ Transcription

#### Option 1: ใช้ Basic Pitch (แนะนำเริ่มต้น)
```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# แปลงเสียงพิณเป็น MIDI
model_output, midi_data, note_events = predict(
    "audio/phin_sample.wav"
)

# บันทึก MIDI
midi_data.write("output/phin_transcription.mid")

# วิเคราะห์ note events
for note in note_events:
    print(f"Pitch: {note['pitch']}, "
          f"Start: {note['start_time']}, "
          f"Duration: {note['duration']}")
```

#### Option 2: ใช้วิธี KMUTT (Energy + Spectral)
```python
# Onset Detection (EWMA)
def detect_onsets_ewma(audio, threshold=0.3):
    # Energy calculation
    energy = librosa.feature.rms(y=audio)[0]
    
    # EWMA smoothing
    alpha = 0.1
    smoothed = np.zeros_like(energy)
    smoothed[0] = energy[0]
    for i in range(1, len(energy)):
        smoothed[i] = alpha * energy[i] + (1 - alpha) * smoothed[i-1]
    
    # Peak detection
    onsets = librosa.util.peak_pick(
        smoothed, 
        pre_max=3, 
        post_max=3, 
        pre_avg=3, 
        post_avg=5, 
        delta=threshold, 
        wait=10
    )
    return onsets

# Pitch Detection (FFT)
def detect_pitch_fft(audio_segment, sr=22050):
    # FFT
    fft = np.fft.fft(audio_segment)
    magnitude = np.abs(fft)
    
    # Find peak
    peak_idx = np.argmax(magnitude)
    frequency = peak_idx * sr / len(audio_segment)
    
    return frequency
```

### การประเมินผล

#### Metrics สำหรับ Classification
```python
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix
)

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)

# Per-class metrics
conf_matrix = confusion_matrix(y_true, y_pred)
```

#### Metrics สำหรับ Transcription
```python
import mir_eval

# Onset detection
onset_precision, onset_recall, onset_f1 = mir_eval.onset.f_measure(
    reference_onsets, estimated_onsets, window=0.05
)

# Note transcription
note_precision, note_recall, note_f1 = mir_eval.transcription.precision_recall_f1_overlap(
    reference_intervals, reference_pitches,
    estimated_intervals, estimated_pitches
)

# Pitch accuracy
pitch_accuracy = mir_eval.melody.voicing_measures(
    reference_pitches, estimated_pitches
)
```

---

## 📁 ข้อมูลเทคนิคจากโปรเจคปัจจุบัน

### โครงสร้างโปรเจค
```
kritsanan1-isan-instruments/
├── src/
│   ├── data_collection/
│   │   └── youtube_downloader.py      # ดาวน์โหลดจาก YouTube
│   ├── preprocessing/
│   │   └── audio_processor.py         # โหลด, normalize, segment
│   ├── features/
│   │   └── feature_extractor.py       # MFCC, Chroma, Spectral
│   ├── models/
│   │   ├── classifier.py              # Random Forest
│   │   └── dataset_manager.py         # Metadata management
│   ├── evaluation/
│   │   └── model_evaluator.py         # Metrics, plots
│   └── transcription/
│       ├── music_transcriber.py       # Transcription logic
│       ├── onset_detector.py          # Onset detection
│       └── pitch_detector.py          # Pitch detection
├── data/
│   ├── raw/                           # Original recordings
│   ├── processed/                     # Preprocessed audio
│   └── metadata/                      # JSON metadata
├── models/                            # Trained models
├── docs/
│   ├── METHODOLOGY.md                 # Technical methodology
│   ├── DATA_COLLECTION_PROTOCOL.md    # Ethical guidelines
│   ├── THAI_MUSIC_SYSTEM.md           # Thai music theory
│   └── TRANSCRIPTION_GUIDE.md         # Transcription guide
├── examples/
│   ├── download_youtube_videos.py
│   ├── generate_demo_data.py
│   └── transcribe_demo.py
├── app.py                             # Streamlit web app
└── train_model.py                     # Training pipeline
```

### Dependencies
```python
# pyproject.toml
dependencies = [
    "joblib>=1.5.2",
    "librosa>=0.11.0",
    "matplotlib>=3.10.7",
    "mir-eval>=0.8.2",
    "numpy>=2.3.5",
    "pandas>=2.3.3",
    "plotly>=6.5.0",
    "pretty-midi>=0.2.11",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "soundfile>=0.13.1",
    "streamlit>=1.51.0",
    "yt-dlp>=2025.11.12",
]
```

### Features Extracted (143 total)

#### 1. MFCCs (80 features)
- 20 coefficients
- 4 statistics each (mean, std, max, min)
- Captures timbral characteristics

#### 2. Chroma Features (48 features)
- 12 bins (musical notes)
- 4 statistics each
- Represents pitch content

#### 3. Spectral Features (12 features)
- Spectral Centroid (brightness)
- Spectral Rolloff (frequency distribution)
- Spectral Bandwidth (frequency range)
- 4 statistics each

#### 4. Temporal Features (2 features)
- Zero-crossing rate
- RMS energy

#### 5. Pitch Features (1 feature)
- Fundamental frequency

### Ethical Framework

#### Consent Protocol
```python
# dataset_manager.py
dataset_manager.add_recording(
    file_path="audio/phin_sample.wav",
    instrument="Phin",
    technique="Traditional",
    performer_name="นายทองใส ทับถนน",
    consent_status=True,              # Required!
    cultural_attribution="ศิลปินอีสาน",
    notes="ลายลำเพลิน"
)

# Validate consent before training
consent_stats = dataset_manager.validate_consent()
if consent_stats['consent_rate'] < 1.0:
    print("⚠️ Not all recordings have consent!")
```

#### Cultural Respect Guidelines
1. ✅ Documented performer consent required
2. ✅ Proper cultural attribution
3. ✅ Transparent methodology
4. ✅ Community engagement
5. ✅ Maintain cultural context
6. ❌ No commercial exploitation without consent
7. ❌ No cultural appropriation
8. ❌ No misrepresentation of traditions

---

## 🎯 แผนการดำเนินงานแนะนำ

### Phase 1: Data Collection (2-4 สัปดาห์)
1. ✅ รวบรวม YouTube links (ทำแล้ว - มีรายการด้านบน)
2. ⏳ ดาวน์โหลดวิดีโอ/เสียง (ใช้ yt-dlp)
3. ⏳ แยกเสียงเป็น WAV 22,050 Hz
4. ⏳ สร้าง metadata (ชื่อ, ลาย, ศิลปิน)
5. ⏳ หาข้อมูล consent (ติดต่อศิลปิน/ช่อง)

**Target:** 100-200 samples ต่อเครื่องดนตรี

### Phase 2: Preprocessing (1 สัปดาห์)
1. ⏳ Audio quality check
2. ⏳ Normalization
3. ⏳ Segmentation (3 sec, 50% overlap)
4. ⏳ Data augmentation (optional)
   - Pitch shifting (±2 semitones)
   - Time stretching (0.9-1.1×)
   - Add noise

### Phase 3: Feature Extraction (1 สัปดาห์)
1. ⏳ Extract MFCC, Chroma, Spectral
2. ⏳ Verify feature quality
3. ⏳ Save feature vectors
4. ⏳ Create training/validation split

### Phase 4: Model Training (1-2 สัปดาห์)
1. ⏳ Train baseline Random Forest
2. ⏳ Cross-validation
3. ⏳ Hyperparameter tuning
4. ⏳ Experiment with other models:
   - SVM
   - XGBoost
   - Neural Networks (CNN)

### Phase 5: Transcription (Optional, 2-3 สัปดาห์)
1. ⏳ Test Basic Pitch on Thai music
2. ⏳ Fine-tune if needed
3. ⏳ Implement onset/pitch detection
4. ⏳ Validate against โน๊ตลายพิณ

### Phase 6: Evaluation & Documentation (1 สัปดาห์)
1. ⏳ Comprehensive testing
2. ⏳ Error analysis
3. ⏳ Documentation
4. ⏳ Demo preparation

---

## 📚 แหล่งข้อมูลเพิ่มเติม

### Datasets Download Links
- **NSynth:** https://magenta.withgoogle.com/datasets/nsynth
- **OpenMIC-2018:** https://github.com/cosmir/openmic-2018
- **FMA:** https://github.com/mdeff/fma
- **MusicNet:** https://benadar293.github.io/
- **Slakh2100:** https://github.com/KinWaiCheuk/slakh_loader

### Tools & Libraries
- **Basic Pitch:** https://github.com/spotify/basic-pitch
- **Omnizart:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
- **librosa:** https://librosa.org/
- **mir_eval:** https://craffel.github.io/mir_eval/
- **pretty_midi:** https://craffel.github.io/pretty-midi/

### Research Papers
- **Basic Pitch Paper:** https://arxiv.org/abs/2203.09893
- **MT3:** https://arxiv.org/abs/2111.03017
- **YourMT3+:** https://arxiv.org/html/2407.04822v1
- **Thai Music System:** https://journal.iftawm.org/wp-content/uploads/2022/02/Garzoli_AAWM_Vol_4_2.pdf

### YouTube Playlists/Channels
- **มูนมังอีสาน:** สอนพิณพื้นฐาน
- **ดุลย์เพลงพิณ:** ลายพิณต่าง ๆ
- **สตีฟ ฐิติวัสส์:** โน๊ตพิณเยอะ
- **M MUSIC GROUP:** สอนแบบมีโน๊ต

---

## 🔗 Next Steps

### ทันที (สัปดาห์นี้)
1. ✅ ดาวน์โหลดโน๊ตพิณจาก guitar285.wordpress.com
2. ⏳ ศึกษา Basic Pitch และทดลองกับเสียงพิณ
3. ⏳ ดาวน์โหลด FMA small subset สำหรับทดสอบ
4. ⏳ เริ่มรวบรวม YouTube links อย่างเป็นระบบ

### สัปดาห์หน้า
1. ⏳ ติดตั้งและทดลอง yt-dlp
2. ⏳ ดาวน์โหลดเสียงตัวอย่าง 10-20 ไฟล์
3. ⏳ ทดลอง pipeline ใน project ปัจจุบัน
4. ⏳ ตั้งค่า AI Drive สำหรับเก็บข้อมูล

### เดือนหน้า
1. ⏳ รวบรวมข้อมูลให้ครบ 100-200 samples
2. ⏳ Train baseline model
3. ⏳ Evaluate และปรับปรุง
4. ⏳ เริ่มทำ transcription experiments

---

## 📞 ข้อมูลติดต่อ

### Project Repository
- **GitHub:** kritsanan1/isan-instruments

### Useful Communities
- **ISMIR (International Society for Music Information Retrieval)**
- **Magenta Community**
- **Music Technology StackExchange**

---

**หมายเหตุ:** เอกสารนี้รวบรวมข้อมูลจากการสืบค้นเชิงลึกเมื่อวันที่ 24 พฤศจิกายน 2025 
สำหรับโปรเจค Isan Musical Instruments Classifier โดยคำนึงถึงความเคารพต่อวัฒนธรรมและจริยธรรม AI

**สร้างโดย:** AI Research Assistant
**วัตถุประสงค์:** การอนุรักษ์และส่งเสริมดนตรีพื้นบ้านอีสานผ่านเทคโนโลยี AI
