"""Sample-rate conversion."""

from __future__ import annotations

import librosa
import numpy as np


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(
        np.float32
    )
