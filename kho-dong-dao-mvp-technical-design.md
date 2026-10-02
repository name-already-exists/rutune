# Technical Design — MVP Reference-Guided Singing Voice Processing

## 1. Tổng quan

### 1.1. Mục tiêu

MVP xây dựng một audio processing engine có khả năng nhận:

- **Bài hát mẫu / Reference**
- **Bản người dùng tự hát / User Vocal**

Sau đó xử lý bản hát của người dùng để:

- Giai điệu gần với bài mẫu.
- Cao độ từng câu/nốt gần với bài mẫu.
- Timing và nhịp hát gần với bài mẫu.
- Độ dài từng nốt/câu gần với bài mẫu.
- Khoảng ngắt giữa các câu gần với bài mẫu.
- Vẫn giữ **chất giọng, màu giọng và đặc trưng giọng hát của người dùng**.

### 1.2. Positioning của MVP

> Bài mẫu quyết định **cách thể hiện**.  
> Giọng người dùng quyết định **ai đang hát**.

MVP không nhằm tạo một ca sĩ mới hoặc thay toàn bộ giọng người dùng bằng giọng của ca sĩ trong bài mẫu.

### 1.3. Phạm vi MVP

#### Input

```text
reference.wav
user_vocal.wav
```

Khuyến nghị:

- Reference nên là vocal sạch hoặc vocal stem.
- User audio nên là một bản thu giọng hát tương đối sạch.
- Mono hoặc stereo đều có thể hỗ trợ, nhưng engine nên convert về mono cho bước phân tích.
- WAV/PCM là format ưu tiên trong pipeline xử lý.

#### Output

```text
processed_vocal.wav
```

Có thể mở rộng:

```text
processed_vocal.mp3
processed_vocal_with_music.mp3
```

---

# 2. Mục tiêu kỹ thuật

## 2.1. Những yếu tố cần transfer từ Reference

| Thành phần | Reference → User | Mục tiêu |
|---|---:|---|
| Melody | Có | Cao |
| Pitch contour | Có | Cao |
| Timing | Có | Cao |
| Note duration | Có | Cao |
| Phrase duration | Có | Cao |
| Pause | Có | Trung bình/Cao |
| Dynamics | Có | Trung bình |
| Timbre | Không | Giữ user |
| Voice identity | Không | Giữ user |
| Lyrics | Không bắt buộc | Không cần cho audio engine |

Điểm quan trọng:

**Không copy giọng của reference.**

Chỉ transfer các đặc tính liên quan đến performance.

---

# 3. Kiến trúc tổng thể

## 3.1. MVP local processing

Phiên bản đầu tiên không cần backend phức tạp.

```text
                  ┌─────────────────────┐
                  │   Reference Audio   │
                  │    reference.wav    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Reference Analyzer  │
                  │                     │
                  │ - F0                │
                  │ - Melody            │
                  │ - Onset             │
                  │ - Phrase            │
                  │ - Silence            │
                  │ - Timing             │
                  └──────────┬──────────┘
                             │
                             ▼
                       Reference Map
                             │
                             │
┌──────────────────────┐     │
│    User Vocal       │     │
│    user.wav         │     │
└──────────┬───────────┘     │
           │                 │
           ▼                 │
┌─────────────────────┐      │
│   User Analyzer     │      │
│                     │      │
│ - F0                │      │
│ - Onset             │      │
│ - Silence           │      │
│ - Energy            │      │
└──────────┬──────────┘      │
           │                 │
           └────────┬────────┘
                    ▼
             ┌───────────────┐
             │ Alignment     │
             │ DTW           │
             └───────┬───────┘
                     │
                     ▼
             Performance Map
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Pitch Correction       Time Alignment
          │                     │
          └──────────┬──────────┘
                     ▼
              Voice Preserve
                     │
                     ▼
             Post Processing
                     │
                     ▼
              processed.wav
```

---

# 4. Công nghệ

## 4.1. Python

Khuyến nghị:

```text
Python 3.11+
```

Python được sử dụng làm audio processing engine.

---

## 4.2. Core libraries

### librosa

Dùng cho:

- Audio loading.
- STFT.
- Mel spectrogram.
- Chroma.
- Onset detection.
- Beat analysis.
- Audio feature extraction.
- `pyin` F0 estimation trong baseline.

```bash
pip install librosa
```

---

### NumPy

Dùng cho:

- Audio array.
- F0 array.
- Signal processing.
- Mathematical transformations.

```bash
pip install numpy
```

---

### SciPy

Dùng cho:

- Filtering.
- Smoothing.
- Signal processing.
- Interpolation.

```bash
pip install scipy
```

---

### SoundFile

Dùng để đọc/ghi WAV.

```bash
pip install soundfile
```

---

### torchcrepe / CREPE

Dùng để estimate F0.

F0 là thành phần rất quan trọng trong MVP vì hệ thống cần biết:

```text
User đang hát nốt nào?
Reference đang hát nốt nào?
```

Khuyến nghị thử:

```bash
pip install torchcrepe
```

Nếu môi trường chưa cần GPU thì có thể chạy CPU cho prototype.

---

### Rubber Band

Dùng cho:

- Time stretching.
- Pitch shifting.

Rubber Band phù hợp hơn các phép resampling đơn giản khi cần giữ chất lượng vocal sau khi thay đổi pitch/time.

Python wrapper có thể sử dụng:

```bash
pip install pyrubberband
```

Cần cài Rubber Band CLI ở môi trường runtime.

---

### Pedalboard

Dùng cho post-processing:

- EQ.
- Compressor.
- Limiter.
- Reverb.
- Gain.

```bash
pip install pedalboard
```

Pedalboard **không phải engine chính để transfer melody**.

---

### FFmpeg

Dùng cho:

- MP3/AAC/WAV conversion.
- Resampling.
- Format normalization.
- Audio muxing.

---

# 5. Audio preprocessing

Trước khi phân tích cần đưa cả hai audio về cùng chuẩn.

## 5.1. Normalize

Ví dụ:

```text
Sample Rate: 44.1 kHz hoặc 48 kHz
Bit depth: 24-bit/32-bit float
Channels: mono cho analysis
```

Pipeline:

```text
Input
  ↓
Decode
  ↓
Resample
  ↓
Convert mono
  ↓
Peak/RMS normalization
  ↓
Analysis
```

Không nên normalize quá mạnh vì có thể làm thay đổi dynamics.

---

# 6. Voice Activity Detection

Cần xác định:

```text
Singing
Silence
Breath
Noise
```

MVP có thể bắt đầu bằng energy-based detection.

Ví dụ:

```text
RMS < threshold
    → silence

RMS >= threshold
    → vocal activity
```

Có thể kết hợp:

- RMS
- Spectral energy
- F0 confidence

Một frame được xem là singing nếu:

```text
energy > threshold
AND
F0 confidence > threshold
```

---

# 7. Reference Analysis

Reference Analyzer tạo ra một `ReferenceMap`.

Ví dụ:

```json
{
  "duration": 42.5,
  "phrases": [
    {
      "start": 0.8,
      "end": 4.2,
      "notes": [
        {
          "start": 0.8,
          "end": 1.2,
          "f0": 261.6
        },
        {
          "start": 1.2,
          "end": 1.8,
          "f0": 293.7
        }
      ]
    }
  ]
}
```

---

# 8. F0 / Melody Extraction

## 8.1. F0

F0 là fundamental frequency.

Ví dụ:

```text
C4 ≈ 261.63 Hz
D4 ≈ 293.66 Hz
E4 ≈ 329.63 Hz
```

Reference:

```text
F0_ref[t]
```

User:

```text
F0_user[t]
```

---

## 8.2. Không nên quantize đơn giản

Không nên làm:

```text
User F0
    ↓
nearest musical note
    ↓
snap
```

vì sẽ làm vocal có cảm giác:

- robotic
- autotune quá mạnh
- mất vibrato
- mất expression

Thay vào đó:

```text
Reference F0
        ↓
Target contour

User F0
        ↓
Deviation
        ↓
Smooth correction
```

---

# 9. Reference-guided Pitch Correction

Giả sử:

```text
F_ref(t)
F_user(t)
```

Sau khi alignment:

```text
pitch_error(t)
    =
F_ref(t) - F_user(t)
```

Không áp dụng 100%:

```text
correction = pitch_error
```

Mà dùng correction strength:

```text
correction =
    pitch_error * correction_strength
```

Ví dụ:

```text
Natural:
0.30

Balanced:
0.60

Close:
0.85
```

Sau đó:

```text
F_target =
    F_user + correction
```

---

# 10. Pitch correction phải giữ expression

Reference không chỉ chứa note.

Nó còn có:

- Vibrato.
- Pitch slide.
- Attack.
- Release.
- Micro pitch variation.

Do đó target F0 nên được smoothing nhưng không flatten hoàn toàn.

Ví dụ:

```text
Reference:

      /\
     /  \
____/    \____
```

Không nên biến thành:

```text
──────────────
```

Mà user nên được đưa gần reference:

```text
User:

    /\__
___/    \____

Target:

     /\
____/  \____
```

---

# 11. Alignment

Đây là một trong những phần quan trọng nhất.

Hai người có thể hát cùng một bài nhưng:

```text
Reference:
|----note----|---note---|

User:
|------note------|--note--|
```

Do đó không thể so sánh frame-to-frame trực tiếp.

---

# 12. Dynamic Time Warping — DTW

DTW được sử dụng để tìm mapping:

```text
User timeline
       ↓
Reference timeline
```

Ví dụ:

```text
User:

A B C D E F
| | | | | |

Reference:

A B C D E F
| | | | | |
```

Mapping có thể là:

```text
user[0] → ref[0]
user[1] → ref[1]
user[2] → ref[3]
user[3] → ref[4]
...
```

Feature có thể dùng:

```text
F0
+
Chroma
+
Energy
```

Không nên phụ thuộc duy nhất vào F0 vì các đoạn:

- silence
- breath
- unvoiced consonant

không có F0 đáng tin cậy.

---

# 13. Phrase Detection

Cần chia bài thành các phrase.

Ví dụ:

```text
Phrase 1
0.5s → 4.2s

Pause
4.2s → 4.8s

Phrase 2
4.8s → 8.9s
```

Reference:

```text
P1 = 3.7s
Pause = 0.6s
P2 = 4.1s
```

User:

```text
P1 = 4.4s
Pause = 0.2s
P2 = 3.8s
```

MVP có thể điều chỉnh:

```text
P1 → 3.7s
Pause → 0.6s
P2 → 4.1s
```

---

# 14. Time Stretching

Sau alignment, từng phrase có thể được time-stretch.

Ví dụ:

```text
User phrase:
4.4 seconds

Reference:
3.7 seconds
```

Scale:

```text
3.7 / 4.4 = 0.84
```

Dùng Rubber Band để giảm duration.

Không nên thay đổi toàn bộ bài bằng một hệ số duy nhất.

Nên xử lý theo:

```text
Phrase
  ↓
Note/segment
  ↓
Time stretch
```

để giữ cách thể hiện tự nhiên hơn.

---

# 15. Pause Matching

Pause là khoảng thời gian không có singing activity.

Có thể phát hiện:

```text
RMS < threshold
```

Sau đó xác định:

```text
pause_start
pause_end
pause_duration
```

Reference:

```text
Pause = 0.75s
```

User:

```text
Pause = 0.25s
```

Có thể thêm:

```text
silence padding
```

hoặc time-stretch vùng phrase trước/sau.

### Lưu ý

Không nên ép pause tuyệt đối trong mọi trường hợp.

Pause nên có:

```text
soft constraint
```

để tránh kết quả quá máy móc.

---

# 16. Voice Preservation

Đây là yêu cầu quan trọng nhất của project.

Có hai nhóm thông tin:

### Singer Identity

```text
Timbre
Formant
Resonance
Harmonic structure
Voice characteristics
```

### Performance

```text
Pitch
Timing
Duration
Pause
Dynamics
```

Mục tiêu:

```text
Performance
    → gần Reference

Voice Identity
    → giữ User
```

---

# 17. Không dùng Voice Conversion ở MVP

MVP **không cần**:

```text
User Voice
    ↓
AI Voice Conversion
    ↓
Reference Singer Voice
```

Vì đây là hướng khác.

MVP cần:

```text
User Voice
    ↓
Pitch / Time / Performance correction
    ↓
Same User Voice
```

Điều này giúp scope nhỏ hơn rất nhiều.

---

# 18. Processing Pipeline

Pipeline đề xuất:

```text
1. Load audio
        ↓
2. Normalize audio
        ↓
3. Detect vocal activity
        ↓
4. Extract F0
        ↓
5. Extract energy
        ↓
6. Detect phrase / silence
        ↓
7. Analyze Reference
        ↓
8. Analyze User
        ↓
9. DTW alignment
        ↓
10. Build performance mapping
        ↓
11. Calculate target F0
        ↓
12. Smooth pitch target
        ↓
13. Pitch correction
        ↓
14. Time alignment
        ↓
15. Pause adjustment
        ↓
16. Dynamics processing
        ↓
17. EQ
        ↓
18. Compression
        ↓
19. Limiter
        ↓
20. Export
```

---

# 19. Project Structure

Đề xuất:

```text
singing-processor/
│
├── app/
│   ├── __init__.py
│   │
│   ├── config.py
│   │
│   ├── audio/
│   │   ├── loader.py
│   │   ├── normalize.py
│   │   ├── resample.py
│   │   ├── silence.py
│   │   └── export.py
│   │
│   ├── analysis/
│   │   ├── f0.py
│   │   ├── energy.py
│   │   ├── onset.py
│   │   ├── phrase.py
│   │   └── reference.py
│   │
│   ├── alignment/
│   │   ├── dtw.py
│   │   └── mapping.py
│   │
│   ├── processing/
│   │   ├── pitch.py
│   │   ├── timing.py
│   │   ├── pause.py
│   │   └── dynamics.py
│   │
│   ├── effects/
│   │   ├── eq.py
│   │   ├── compressor.py
│   │   └── limiter.py
│   │
│   ├── pipeline/
│   │   └── processor.py
│   │
│   └── cli.py
│
├── tests/
│   ├── test_f0.py
│   ├── test_alignment.py
│   ├── test_pitch.py
│   ├── test_timing.py
│   └── test_pipeline.py
│
├── samples/
│   ├── reference/
│   ├── input/
│   └── output/
│
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

# 20. Core Data Model

## AudioFeatures

```python
@dataclass
class AudioFeatures:
    sample_rate: int
    duration: float
    f0: np.ndarray
    f0_confidence: np.ndarray
    energy: np.ndarray
    timestamps: np.ndarray
```

## Phrase

```python
@dataclass
class Phrase:
    start: float
    end: float
    confidence: float
```

## ReferenceMap

```python
@dataclass
class ReferenceMap:
    duration: float
    f0: np.ndarray
    timestamps: np.ndarray
    phrases: list[Phrase]
```

## AlignmentMap

```python
@dataclass
class AlignmentMap:
    user_indices: np.ndarray
    reference_indices: np.ndarray
    distance: float
```

---

# 21. Configuration

MVP nên cho phép cấu hình:

```yaml
audio:
  sample_rate: 44100
  channels: 1

pitch:
  correction_strength: 0.70
  max_shift_semitones: 4
  smoothing: 0.15

timing:
  enabled: true
  strength: 0.70

pause:
  enabled: true
  strength: 0.60

dynamics:
  compression: true
  limiter: true

output:
  format: wav
```

---

# 22. Correction Strength

Người dùng không cần nhìn thấy các thông số kỹ thuật.

Có thể expose thành:

```text
Natural
Balanced
Close to Sample
```

Mapping:

```text
Natural
→ pitch 0.35
→ timing 0.30
→ pause 0.25

Balanced
→ pitch 0.60
→ timing 0.55
→ pause 0.50

Close to Sample
→ pitch 0.85
→ timing 0.80
→ pause 0.75
```

Các giá trị này chỉ là starting point cho thử nghiệm, không phải thông số cố định.

---

# 23. MVP CLI

Có thể chạy:

```bash
python -m app.cli \
  --reference samples/reference/song.wav \
  --input samples/input/user.wav \
  --output samples/output/result.wav
```

Hoặc:

```bash
python -m app.cli \
  --reference reference.wav \
  --input user.wav \
  --output result.wav \
  --strength balanced
```

---

# 24. API Layer — chưa bắt buộc

Sau khi core processor chạy ổn định, có thể thêm FastAPI.

```text
POST /api/v1/process
```

Request:

```multipart
reference
user_audio
strength
```

Response:

```json
{
  "jobId": "abc123",
  "status": "PROCESSING"
}
```

Status:

```http
GET /api/v1/process/{jobId}
```

Response:

```json
{
  "jobId": "abc123",
  "status": "COMPLETED",
  "outputUrl": "/files/abc123.wav"
}
```

---

# 25. Kiến trúc khi đưa vào sản phẩm

Khi MVP local đã ổn:

```text
Mobile / Web
      │
      ▼
Spring Boot API
      │
      ▼
Job Queue
      │
      ▼
Python Audio Worker
      │
      ├── Reference Analyzer
      ├── User Analyzer
      ├── Alignment
      ├── Pitch Processor
      ├── Timing Processor
      └── Post Processor
      │
      ▼
Object Storage
      │
      ▼
Processed Audio
```

Spring Boot có thể quản lý:

- Authentication.
- User.
- Song catalog.
- Upload.
- Job.
- Processing status.
- Download.
- History.

Python chỉ chịu trách nhiệm:

```text
Audio Processing
```

---

# 26. Không đưa AI API vào MVP

MVP đầu tiên cố tình không dùng:

```text
OpenAI Audio API
ElevenLabs
Kits AI
Suno
Voice Conversion API
```

Lý do:

1. Kiểm soát thuật toán.
2. Không phụ thuộc vendor.
3. Dễ benchmark.
4. Biết chính xác phần nào đang hoạt động.
5. Giảm chi phí thử nghiệm.
6. Có thể thay thế từng component sau này.

AI có thể được đưa vào Phase 2/3 khi xác định rõ điểm nào DSP chưa giải quyết tốt.

---

# 27. Những phần DSP có thể giải quyết tốt

MVP có thể tập trung vào:

### Pitch

```text
User F0
→ Reference F0
→ Pitch correction
```

### Timing

```text
User phrase duration
→ Reference phrase duration
```

### Pause

```text
Reference pause
→ User pause
```

### Dynamics

```text
Gain
→ Compressor
→ Limiter
```

### Tone

```text
EQ
```

---

# 28. Những phần khó

Không nên kỳ vọng MVP giải quyết hoàn hảo:

### 28.1. Polyphonic reference

Nếu reference chứa:

```text
Vocal
+
Piano
+
Guitar
+
Drums
```

F0 extraction có thể bị nhiễu.

Giải pháp tốt hơn:

```text
Full Song
    ↓
Vocal Separation
    ↓
Reference Vocal
    ↓
Analysis
```

Nhưng vocal separation có thể để Phase 2.

---

### 28.2. User hát sai quá nhiều

Ví dụ reference:

```text
C4 → D4 → E4
```

User:

```text
C4 → A3 → F3
```

Việc ép user lên E4 bằng pitch shifting lớn có thể làm chất lượng giọng giảm.

MVP nên giới hạn:

```text
max_shift_semitones
```

và đánh dấu những đoạn correction quá lớn.

---

### 28.3. Khác octave

Reference:

```text
C4
```

User:

```text
C3
```

Cần detect octave relationship trước khi correction.

Không nên mặc định coi đây là sai pitch.

---

### 28.4. Khác cách hát

Ví dụ:

```text
Reference:
ngân dài

User:
hát ngắn
```

Việc kéo dài user vocal quá nhiều có thể tạo artifact.

Do đó time stretch cần giới hạn ratio.

---

# 29. Artifact Detection

Sau processing cần kiểm tra:

```text
pitch shift quá lớn
time stretch quá lớn
F0 confidence thấp
silence bất thường
clipping
```

Ví dụ:

```python
if abs(pitch_shift) > 4:
    flag("LARGE_PITCH_SHIFT")

if time_ratio > 1.5:
    flag("LARGE_TIME_STRETCH")

if clipping_detected:
    flag("CLIPPING")
```

---

# 30. Quality Metrics

MVP cần benchmark bằng dữ liệu thật.

## Pitch Accuracy

So sánh:

```text
Processed F0
vs
Reference F0
```

Metric:

```text
Mean Absolute Error
```

hoặc:

```text
cents error
```

---

## Timing Accuracy

So sánh:

```text
Phrase start
Phrase end
Note duration
Pause duration
```

---

## Voice Preservation

Cần đo riêng:

```text
Original User
vs
Processed User
```

Mục tiêu:

```text
Voice similarity cao
```

trong khi:

```text
Reference similarity về performance cao
```

Không nên đánh giá chỉ bằng similarity với reference.

---

# 31. Test Dataset

MVP nên xây dựng dataset nội bộ nhỏ:

```text
10 bài hát
×
3–5 người hát
×
2–3 kiểu hát
```

Ví dụ:

```text
Dataset:
30–50 recordings
```

Mỗi recording lưu:

```text
reference.wav
user.wav
expected.wav
```

Trong đó `expected.wav` có thể được đánh giá thủ công ở giai đoạn đầu.

---

# 32. A/B Evaluation

Mỗi test case:

```text
A = Original User

B = Processed User

R = Reference
```

Người test đánh giá:

```text
1. Melody similarity to R
2. Timing similarity to R
3. Pause similarity to R
4. Voice identity preservation
5. Naturalness
```

Không nên chỉ hỏi:

```text
"Nghe hay hơn không?"
```

vì mục tiêu MVP là:

```text
Closer performance
+
Same singer identity
```

---

# 33. Development Roadmap

## Phase 0 — Audio Playground

Mục tiêu:

```text
Load
→ visualize waveform
→ visualize spectrogram
→ extract F0
→ play audio
```

Deliverable:

```text
reference_analysis.png
user_analysis.png
```

---

## Phase 1 — Pitch

Implement:

```text
Reference F0
User F0
DTW
Target F0
Pitch correction
```

Output:

```text
user_pitch_corrected.wav
```

Success criteria:

- Pitch gần reference hơn.
- Voice vẫn nhận ra là user.
- Không có artifact nghiêm trọng.

---

## Phase 2 — Timing

Thêm:

```text
Phrase detection
DTW
Time stretching
```

Output:

```text
user_pitch_timing_corrected.wav
```

---

## Phase 3 — Pause

Thêm:

```text
Silence detection
Pause mapping
Phrase boundary alignment
```

Output:

```text
user_performance_corrected.wav
```

---

## Phase 4 — Audio Polish

Thêm:

```text
EQ
Compression
Limiter
Loudness normalization
```

Output:

```text
final.wav
```

---

# 34. MVP Definition of Done

MVP được xem là đạt khi:

### Input

```text
Reference vocal
+
User singing
```

### Output

```text
Processed user singing
```

### Output phải đạt

```text
✓ Melody gần reference hơn
✓ Pitch gần reference hơn
✓ Timing gần reference hơn
✓ Phrase duration gần reference hơn
✓ Pause gần reference hơn
✓ Không thay đổi identity của user
✓ Artifact ở mức chấp nhận được
✓ Có thể xử lý tự động từ đầu đến cuối
```

---

# 35. Những thứ KHÔNG nằm trong MVP

Không implement ngay:

```text
❌ Text-to-Speech
❌ Lyric generation
❌ AI song generation
❌ Voice cloning
❌ Voice conversion sang ca sĩ khác
❌ Music generation
❌ Automatic songwriting
❌ Automatic accompaniment generation
❌ Full song separation
❌ Mobile app
❌ User authentication
❌ Payment
❌ Recommendation system
```

Các phần này có thể được xây dựng sau khi core audio engine chứng minh được chất lượng.

---

# 36. Phase 2 — AI Enhancement

Sau khi DSP baseline ổn định, AI có thể được đưa vào những điểm thật sự cần thiết.

Ví dụ:

```text
DSP
 ↓
Pitch / Timing correction
 ↓
AI Naturalization
 ↓
DSP mastering
```

AI có thể hỗ trợ:

- Natural pitch correction.
- Voice preservation.
- Better phrase transformation.
- Better handling of large pitch shifts.
- Artifact reduction.
- Breath/expression preservation.

Điểm quan trọng:

**AI nên nâng cấp audio engine, không thay thế mục tiêu "giữ giọng người dùng".**

---

# 37. Product Principle

Core principle của project:

```text
REFERENCE
    ↓
"Bạn nên hát như thế nào?"
    ↓
Performance guidance

USER VOICE
    ↓
"Ai đang hát?"
    ↓
Voice identity
```

Kết quả:

```text
                 Reference
                    │
             ┌──────┴──────┐
             │ Performance │
             │   Transfer  │
             └──────┬──────┘
                    │
                    ▼
             User's Voice
                    │
                    ▼
             Final Singing
```

Hay nói ngắn gọn:

> **Reference quyết định cách hát. User quyết định giọng hát.**

---

# 38. Kết luận

MVP không cần bắt đầu bằng một mô hình AI lớn.

Một pipeline Python/DSP có thể xây dựng được phiên bản đầu tiên:

```text
Reference
    +
User Singing
    ↓
F0 Analysis
    ↓
Phrase Analysis
    ↓
DTW Alignment
    ↓
Pitch Correction
    ↓
Time Alignment
    ↓
Pause Adjustment
    ↓
Audio Post Processing
    ↓
Final User Singing
```

Kiến trúc này tạo một baseline rõ ràng để đo lường trước khi quyết định có cần AI hay không.

Nếu DSP đã giải quyết tốt 60–80% bài toán, AI chỉ cần tập trung vào những phần còn khó thay vì xây dựng toàn bộ sản phẩm phụ thuộc vào một API bên ngoài.
