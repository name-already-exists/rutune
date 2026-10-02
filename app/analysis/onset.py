"""Onset detection."""

from __future__ import annotations

import librosa
import numpy as np


def detect_onsets(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Onset times in seconds."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop_length, backtrack=False, units="frames"
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
