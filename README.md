# Rutune

Engine xử lý giọng hát theo bài mẫu (MVP, DSP thuần — xem `kho-dong-dao-mvp-technical-design.md`).

> Reference quyết định **cách hát**. User quyết định **giọng hát**.

Trạng thái: **Phase 0 + Phase 1** (phân tích, alignment, pitch correction). Timing, pause,
dynamics và polish (Phase 2–4) chưa áp dụng.

## Cài đặt (Windows)

```powershell
py install 3.12                      # numba/torch chưa hỗ trợ ổn định Python 3.14
py -V:3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
# tuỳ chọn: .[crepe] (F0 bằng torchcrepe), .[rubberband], .[speaker] (ECAPA similarity)
.\.venv\Scripts\python.exe -m app.cli doctor
```

Công cụ ngoài:

- **ffmpeg** — cần để đọc mp3/m4a và xuất mp3 (`winget install Gyan.FFmpeg`).
- **rubberband** CLI v3 — chỉ cần cho `--backend rubberband` (tải từ breakfastquay.com, thêm vào PATH).

## Sử dụng

```powershell
# Phase 0: biểu đồ phân tích (waveform, spectrogram, F0, phrase/note, breath)
python -m app.cli analyze samples/reference/song.wav samples/input/user.wav -o samples/output

# Phase 1: pitch correction theo reference
python -m app.cli process `
  -r samples/reference/song.wav `
  -i samples/input/user.wav `
  -o samples/output/user_pitch_corrected.wav `
  --strength balanced --debug-dir samples/output/debug
```

Kết quả:

- `user_pitch_corrected.wav` — giọng user đã chỉnh cao độ (timeline giữ nguyên).
- `user_pitch_corrected.report.json` — alignment (cost ratio, offset), octave shift,
  metric pitch trước/sau, độ giữ giọng (MCD, LTAS), các cờ (`LARGE_PITCH_SHIFT`,
  `KEY_MISMATCH`, `LOW_F0_CONFIDENCE`, `CLIPPING`, `ALIGNMENT_FAILED`...).
- `--debug-dir`: `pitch_correction.png` (user / reference / target / processed),
  `alignment.png`, `*_analysis.png`, `reference_map.json`, `curves.npz`.

Strength preset: `natural` (0.35) · `balanced` (0.60) · `close` (0.85). Các tham số khác
chỉnh qua `--config configs/default.yaml`.

Phân tích reference được cache trong `.cache/analysis/` (theo hash file + cấu hình).

## Cách hoạt động (Phase 1)

1. Load → mono → resample; phân tích ở 16 kHz, hop 10 ms: F0 (pYIN), RMS, centroid, chroma, onset.
2. VAD: sing / unvoiced / breath / silence → phrase → note.
3. Alignment: DTW thô (≥50 ms, tự cắt im lặng, tự chuyển subsequence nếu user chỉ hát một đoạn)
   → DTW mịn theo từng phrase (10 ms). Cost ratio quá cao → `ALIGNMENT_FAILED`.
4. Mapping: F0 reference trên timeline user; tự nhận lệch octave.
5. Correction = (target − user) × strength, giới hạn `max_shift_semitones`, làm mượt **đường
   correction** (không làm mượt F0 → giữ vibrato của user).
6. Resynthesis bằng WORLD: thay F0, giữ spectral envelope (màu giọng / formant).

## License lưu ý

Backend mặc định (WORLD, librosa, scipy) dùng license BSD/MIT/ISC. **Rubber Band** (và
Pedalboard nếu dùng sau này) là GPL — cần license thương mại khi đưa vào sản phẩm.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Test dùng tín hiệu hát tổng hợp có ground truth (cao độ, timing, vibrato, breath), không cần
file thu thật.
