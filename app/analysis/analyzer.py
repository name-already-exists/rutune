"""Full frame-level analysis of one vocal signal (used for both reference and user)."""

from __future__ import annotations

import numpy as np

from app.analysis.energy import chroma, rms_db, spectral_centroid
from app.analysis.f0 import estimate_f0
from app.analysis.onset import detect_onsets
from app.analysis.phrase import build_phrases
from app.audio.silence import mark_breaths, vad_labels
from app.config import Config
from app.models import AudioFeatures


def hop_length(cfg: Config) -> int:
    return int(round(cfg.audio.analysis_sample_rate * cfg.audio.hop_ms / 1000.0))


def analyze_signal(y: np.ndarray, cfg: Config) -> AudioFeatures:
    """Analyse a mono signal sampled at cfg.audio.analysis_sample_rate."""
    sr = cfg.audio.analysis_sample_rate
    hl = hop_length(cfg)
    hop = hl / sr

    f0, voiced, conf = estimate_f0(y, sr, hl, cfg.f0)
    energy = rms_db(y, frame_length=1024, hop_length=hl)
    centroid = spectral_centroid(y, sr, n_fft=1024, hop_length=hl)
    chroma_ = chroma(y, sr, n_fft=2048, hop_length=hl)
    onsets = detect_onsets(y, sr, hl)

    n = min(len(f0), len(energy), len(centroid), chroma_.shape[1])
    f0, voiced, conf = f0[:n], voiced[:n], conf[:n]
    energy, centroid, chroma_ = energy[:n], centroid[:n], chroma_[:, :n]

    labels = vad_labels(energy, voiced, conf, cfg.vad)
    phrases, frames = build_phrases(
        labels, f0, voiced, conf, onsets, hop, cfg.vad, cfg.notes
    )
    in_phrase = np.zeros(n, dtype=bool)
    for s, e in frames:
        in_phrase[s:e] = True
    labels = mark_breaths(labels, centroid, in_phrase, hop, cfg.vad)

    return AudioFeatures(
        sample_rate=sr,
        duration=len(y) / sr,
        hop=hop,
        timestamps=np.arange(n) * hop,
        f0=f0,
        f0_confidence=conf,
        voiced=voiced,
        energy=energy,
        centroid=centroid,
        chroma=chroma_,
        onsets=onsets,
        labels=labels,
        phrases=phrases,
    )
