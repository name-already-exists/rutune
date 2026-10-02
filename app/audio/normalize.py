"""Gentle level normalisation (a single gain, so dynamics are untouched)."""

from __future__ import annotations

import numpy as np


def peak_normalize(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-9:
        return y
    return (y * (10 ** (peak_db / 20.0) / peak)).astype(np.float32)


def remove_dc(y: np.ndarray) -> np.ndarray:
    return (y - np.mean(y)).astype(np.float32)
