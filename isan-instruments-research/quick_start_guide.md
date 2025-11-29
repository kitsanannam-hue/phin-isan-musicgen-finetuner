# 🚀 Quick Start Guide: Isan Instruments Dataset Collection

## เริ่มต้นทันที - สำหรับนักวิจัย

---

## 📥 ขั้นตอนที่ 1: ติดตั้งเครื่องมือ

### ติดตั้ง Dependencies พื้นฐาน
```bash
# Python packages
pip install yt-dlp librosa soundfile numpy pandas

# Basic Pitch (สำหรับ transcription)
pip install basic-pitch

# Optional: mir_eval (สำหรับ evaluation)
pip install mir_eval
```

---

## 🎵 ขั้นตอนที่ 2: ดาวน์โหลดเสียงจาก YouTube

### สร้างไฟล์ YouTube Links
สร้างไฟล์ `youtube_links.txt`:
```
# ลายลำเพลิน
https://www.youtube.com/watch?v=9dERGSNL5Ak
https://www.youtube.com/watch?v=xouLuPjn90A
https://www.youtube.com/watch?v=pKCaf-f19rQ

# ลายแห่
https://www.youtube.com/watch?v=gyDbsN6jbzc
https://www.youtube.com/watch?v=RpSV75Thj4E
https://www.youtube.com/watch?v=HJZxuD57joI

# ลายมโหรีอีสาน
https://www.youtube.com/watch?v=daZpyFy1Qb8
https://www.youtube.com/watch?v=ZT7q9pcWLDc

# สอนพิณพื้นฐาน
https://www.youtube.com/watch?v=ksZ3DWA9mPE
https://www.youtube.com/watch?v=-l1Pj7N_eI8
https://www.youtube.com/watch?v=_Mm7gnrdy08
```

### Script ดาวน์โหลด
สร้างไฟล์ `download_audio.py`:
```python
#!/usr/bin/env python3
"""Download audio from YouTube links for Isan instruments dataset"""
import yt_dlp
from pathlib import Path

def download_audio(url, output_dir="data/raw/phin"):
    """Download audio from YouTube URL"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'postprocessor_args': [
            '-ar', '22050',  # Sample rate
            '-ac', '1',      # Mono
        ],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"✓ Downloaded: {info['title']}")
            return filename.replace('.webm', '.wav').replace('.m4a', '.wav')
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

def batch_download(links_file="youtube_links.txt"):
    """Download multiple URLs from file"""
    with open(links_file, 'r') as f:
        urls = [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]
    
    print(f"Found {len(urls)} URLs to download...")
    
    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Downloading: {url}")
        result = download_audio(url)
        results.append((url, result))
    
    # Summary
    success = sum(1 for _, r in results if r is not None)
    print(f"\n{'='*60}")
    print(f"Download complete: {success}/{len(urls)} successful")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    batch_download()
```

### รันการดาวน์โหลด
```bash
python download_audio.py
```

---

## 🔍 ขั้นตอนที่ 3: ตรวจสอบคุณภาพเสียง

```python
import librosa
import numpy as np
from pathlib import Path

def check_audio_quality(audio_path):
    """Check audio file quality"""
    audio, sr = librosa.load(audio_path, sr=22050)
    
    # Metrics
    duration = len(audio) / sr
    rms_energy = np.sqrt(np.mean(audio**2))
    max_amplitude = np.max(np.abs(audio))
    zero_crossings = np.sum(librosa.zero_crossings(audio))
    
    # Quality checks
    is_valid = True
    issues = []
    
    if duration < 1.0:
        is_valid = False
        issues.append("Too short (< 1 second)")
    
    if rms_energy < 0.001:
        is_valid = False
        issues.append("Too quiet")
    
    if max_amplitude < 0.01:
        is_valid = False
        issues.append("Very low amplitude")
    
    return {
        'valid': is_valid,
        'duration': duration,
        'rms_energy': rms_energy,
        'max_amplitude': max_amplitude,
        'issues': issues
    }

# ตรวจสอบทุกไฟล์
audio_dir = Path("data/raw/phin")
for audio_file in audio_dir.glob("*.wav"):
    quality = check_audio_quality(audio_file)
    status = "✓" if quality['valid'] else "✗"
    print(f"{status} {audio_file.name}")
    if not quality['valid']:
        print(f"  Issues: {', '.join(quality['issues'])}")
```

---

## 📊 ขั้นตอนที่ 4: แปลงเป็น MIDI (Transcription)

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from pathlib import Path

def transcribe_audio(audio_path, output_dir="data/transcriptions"):
    """Transcribe audio to MIDI using Basic Pitch"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Transcribing: {audio_path}")
    
    # Predict
    model_output, midi_data, note_events = predict(audio_path)
    
    # Save MIDI
    midi_path = output_dir / f"{Path(audio_path).stem}.mid"
    midi_data.write(str(midi_path))
    
    print(f"✓ Saved MIDI: {midi_path}")
    print(f"  Notes detected: {len(note_events)}")
    
    # Show first few notes
    for i, note in enumerate(note_events[:5]):
        print(f"  Note {i+1}: Pitch={note['pitch_midi']}, "
              f"Start={note['start_time_s']:.2f}s, "
              f"Duration={note['duration_s']:.2f}s")
    
    return midi_path, note_events

# ทดสอบกับไฟล์หนึ่ง
audio_file = "data/raw/phin/สอนดีดพิณลายลำเพลิน.wav"
transcribe_audio(audio_file)
```

---

## 🏷️ ขั้นตอนที่ 5: สร้าง Metadata

```python
import json
from datetime import datetime
from pathlib import Path

class MetadataManager:
    """Manage dataset metadata"""
    
    def __init__(self, metadata_file="data/metadata/dataset.json"):
        self.metadata_file = Path(metadata_file)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self):
        """Load existing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"recordings": [], "statistics": {}}
    
    def add_recording(self, audio_path, **kwargs):
        """Add recording metadata"""
        record = {
            "id": len(self.data["recordings"]) + 1,
            "file_path": str(audio_path),
            "filename": Path(audio_path).name,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.data["recordings"].append(record)
        self._save()
        return record["id"]
    
    def _save(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# ใช้งาน
manager = MetadataManager()

manager.add_recording(
    audio_path="data/raw/phin/สอนดีดพิณลายลำเพลิน.wav",
    instrument="Phin",
    pattern="ลายลำเพลิน",
    technique="Traditional",
    source="YouTube",
    youtube_url="https://www.youtube.com/watch?v=9dERGSNL5Ak",
    channel="มูนมังอีสาน Channel",
    consent_status="To be obtained",
    notes="สำหรับมือใหม่"
)
```

---

## 📈 ขั้นตอนที่ 6: Extract Features

```python
import librosa
import numpy as np

def extract_features(audio_path):
    """Extract audio features for ML"""
    audio, sr = librosa.load(audio_path, sr=22050)
    
    features = {}
    
    # 1. MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    features['mfcc_max'] = np.max(mfcc, axis=1)
    features['mfcc_min'] = np.min(mfcc, axis=1)
    
    # 2. Chroma (12 bins)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    features['chroma_std'] = np.std(chroma, axis=1)
    
    # 3. Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # 4. Temporal features
    zero_crossings = librosa.zero_crossings(audio)
    features['zero_crossing_rate'] = np.sum(zero_crossings) / len(audio)
    
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = np.mean(rms)
    
    # Flatten to single vector
    feature_vector = []
    for key in sorted(features.keys()):
        val = features[key]
        if isinstance(val, np.ndarray):
            feature_vector.extend(val.tolist())
        else:
            feature_vector.append(val)
    
    return np.array(feature_vector)

# ทดสอบ
features = extract_features("data/raw/phin/สอนดีดพิณลายลำเพลิน.wav")
print(f"Feature vector shape: {features.shape}")
print(f"First 10 features: {features[:10]}")
```

---

## 🤖 ขั้นตอนที่ 7: Train Simple Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# สมมติว่ามี features และ labels แล้ว
X = []  # Feature vectors
y = []  # Labels (Phin, Khaen)

# Load all audio files and extract features
for audio_file in Path("data/raw/phin").glob("*.wav"):
    features = extract_features(audio_file)
    X.append(features)
    y.append("Phin")

for audio_file in Path("data/raw/khaen").glob("*.wav"):
    features = extract_features(audio_file)
    X.append(features)
    y.append("Khaen")

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
top_features = np.argsort(importances)[-10:]
print("\nTop 10 important features:")
for idx in top_features:
    print(f"  Feature {idx}: {importances[idx]:.4f}")
```

---

## 📝 Checklist สำหรับการรวบรวมข้อมูล

### สัปดาห์ที่ 1
- [ ] ติดตั้ง dependencies ทั้งหมด
- [ ] รวบรวม YouTube links (50-100 วิดีโอ)
- [ ] ดาวน์โหลดเสียง 20-30 ไฟล์แรก
- [ ] ตรวจสอบคุณภาพเสียง
- [ ] ทดสอบ Basic Pitch กับเสียงไทย

### สัปดาห์ที่ 2
- [ ] ดาวน์โหลดเสียงให้ครบ 100 ไฟล์
- [ ] สร้าง metadata ทั้งหมด
- [ ] แยกเสียงตามเครื่องดนตรี
- [ ] Transcribe ตัวอย่าง 10-20 ไฟล์

### สัปดาห์ที่ 3
- [ ] Extract features ทั้งหมด
- [ ] สร้าง training/test split
- [ ] Train baseline model
- [ ] Evaluate และวิเคราะห์ผล

### สัปดาห์ที่ 4
- [ ] ปรับปรุงโมเดล
- [ ] เพิ่มข้อมูล (data augmentation)
- [ ] จัดทำเอกสาร
- [ ] เตรียม demo

---

## 🔗 แหล่งข้อมูลเพิ่มเติม

### YouTube Channels สำคัญ
1. **มูนมังอีสาน Channel** - สอนพิณพื้นฐาน
2. **ดุลย์เพลงพิณ** - ลายพิณต่าง ๆ
3. **สตีฟ ฐิติวัสส์ ทองอ่อน** - โน๊ตพิณครบครัน
4. **M MUSIC GROUP** - สอนแบบมีโน๊ต

### Datasets สำรอง
- OpenMIC-2018: https://github.com/cosmir/openmic-2018
- FMA small: https://github.com/mdeff/fma
- NSynth (reference): https://magenta.withgoogle.com/datasets/nsynth

### เอกสารเทคนิค
- Basic Pitch Guide: https://github.com/spotify/basic-pitch
- librosa Tutorial: https://librosa.org/doc/latest/tutorial.html
- scikit-learn Guide: https://scikit-learn.org/stable/tutorial/index.html

---

## 💡 Tips สำหรับความสำเร็จ

### เรื่องเทคนิค
1. ✅ เริ่มจากข้อมูลน้อย แล้วค่อยเพิ่ม
2. ✅ ตรวจสอบคุณภาพเสียงก่อนเทรน
3. ✅ ใช้ validation set เสมอ
4. ✅ บันทึก log ทุกขั้นตอน
5. ✅ backup ข้อมูลสม่ำเสมอ

### เรื่องจริยธรรม
1. ✅ ระบุแหล่งที่มาทุกครั้ง
2. ✅ เคารพลิขสิทธิ์
3. ✅ ไม่ใช้เพื่อการค้าโดยไม่ได้รับอนุญาต
4. ✅ รักษาบริบททางวัฒนธรรม
5. ✅ ติดต่อขอ consent เมื่อเป็นไปได้

---

## 🆘 แก้ปัญหาเบื้องต้น

### ปัญหา: yt-dlp ดาวน์โหลดไม่ได้
```bash
# อัปเดต yt-dlp
pip install --upgrade yt-dlp

# หรือใช้ youtube-dl
pip install youtube-dl
```

### ปัญหา: librosa ขึ้น error
```bash
# ติดตั้ง dependencies เพิ่ม
pip install numba soundfile

# สำหรับ Mac M1
pip install librosa --no-cache-dir
```

### ปัญหา: Memory error
```python
# ประมวลผลทีละไฟล์
for audio_file in audio_files:
    features = extract_features(audio_file)
    # Save immediately
    np.save(f"features/{audio_file.stem}.npy", features)
    # Clear memory
    del features
```

---

**สร้างเมื่อ:** 24 พฤศจิกายน 2025  
**สำหรับ:** Isan Musical Instruments Classifier Project  
**ติดต่อ:** ดูรายละเอียดใน main research document
