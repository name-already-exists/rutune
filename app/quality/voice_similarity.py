"""Voice-identity preservation: original user vs processed user (design §30).

Always computed (no model needed):
- mcd_db: mel-cepstral distortion between time-aligned frames (lower = envelope kept).
- ltas_diff_db: RMS difference of long-term average spectra, 100 Hz-8 kHz.
Optional (pip install -e .[speaker]):
- speaker_cosine: ECAPA speaker-embedding cosine similarity (1.0 = same voice).
"""

from __future__ import annotations

import librosa
import numpy as np
import pyworld as pw

N_CEPS = 25
FRAME_MS = 10.0
_MCD_CONST = 10.0 / np.log(10.0) * np.sqrt(2.0)


def _envelope_cepstrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Mel-cepstrum of the WORLD (pitch-adaptive) spectral envelope + voiced mask.

    Using the envelope rather than plain MFCCs keeps harmonics (i.e. pitch) out of the
    measure, so a pure pitch change with preserved formants scores close to zero.
    """
    x = np.ascontiguousarray(y, dtype=np.float64)
    f0, t = pw.dio(x, sr, f0_floor=65.0, f0_ceil=1100.0, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    return pw.code_spectral_envelope(sp, sr, N_CEPS), f0 > 0


def mel_cepstral_distortion(
    original: np.ndarray, processed: np.ndarray, sr: int, active: np.ndarray | None = None
) -> float:
    """Frame-aligned MCD (dB) over voiced frames, excluding c0. Assumes a shared timeline."""
    a, va = _envelope_cepstrum(original, sr)
    b, vb = _envelope_cepstrum(processed, sr)
    n = min(len(a), len(b))
    mask = va[:n] & vb[:n]
    if active is not None:
        m = min(n, len(active))
        mask[:m] &= active[:m].astype(bool)
        mask[m:] = False
    if not mask.any():
        return float("nan")
    diff = a[:n][mask, 1:] - b[:n][mask, 1:]
    return float(np.mean(_MCD_CONST * np.sqrt(np.sum(diff**2, axis=1))))


def ltas_difference(original: np.ndarray, processed: np.ndarray, sr: int) -> float:
    def ltas(y):
        spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        return 10 * np.log10(spec.mean(axis=1) + 1e-12)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 100) & (freqs <= 8000)
    d = ltas(original)[band] - ltas(processed)[band]
    d -= d.mean()  # ignore overall gain
    return float(np.sqrt(np.mean(d**2)))


def speaker_cosine(original: np.ndarray, processed: np.ndarray, sr: int) -> float | None:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        return None
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=".cache/models/ecapa"
    )
    wavs = [librosa.resample(x, orig_sr=sr, target_sr=16000) for x in (original, processed)]
    with torch.no_grad():
        emb = [model.encode_batch(torch.from_numpy(w)[None]).squeeze() for w in wavs]
    return float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0))


def voice_similarity(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    active: np.ndarray | None = None,
    use_speaker_model: bool = False,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "mcd_db": round(mel_cepstral_distortion(original, processed, sr, active), 2),
        "ltas_diff_db": round(ltas_difference(original, processed, sr), 2),
    }
    if use_speaker_model:
        cos = speaker_cosine(original, processed, sr)
        result["speaker_cosine"] = None if cos is None else round(cos, 4)
    return result
