"""Frame energy and spectral descriptors."""

from __future__ import annotations

import librosa
import numpy as np


def rms_db(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """RMS energy in dB relative to the loudest frame (0 dB = loudest)."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    ref = max(float(rms.max()), 1e-10)
    return 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)


def spectral_centroid(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]


def chroma(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    """L2-normalised chroma (12 x frames); octave-invariant, so robust to octave offsets."""
    c = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, norm=None)
    norm = np.linalg.norm(c, axis=0, keepdims=True)
    return c / np.maximum(norm, 1e-8)
