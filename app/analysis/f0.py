"""F0 estimation (pYIN baseline, optional torchcrepe)."""

from __future__ import annotations

import librosa
import numpy as np

from app.config import F0Config


def estimate_f0(
    y: np.ndarray, sr: int, hop_length: int, cfg: F0Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f0 Hz with NaN when unvoiced, voiced flag, confidence 0..1)."""
    if cfg.method == "pyin":
        return _pyin(y, sr, hop_length, cfg)
    if cfg.method == "crepe":
        return _crepe(y, sr, hop_length, cfg)
    raise ValueError(f"Unknown F0 method '{cfg.method}'")


def _pyin(y, sr, hop_length, cfg):
    f0, voiced, prob = librosa.pyin(
        y,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        sr=sr,
        frame_length=cfg.frame_length,
        hop_length=hop_length,
        center=True,
    )
    f0 = np.where(voiced, f0, np.nan)
    return f0.astype(np.float64), voiced.astype(bool), np.nan_to_num(prob).astype(np.float64)


def _crepe(y, sr, hop_length, cfg):
    try:
        import torch
        import torchcrepe
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("F0 method 'crepe' requires: pip install -e .[crepe]") from exc

    audio = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None]
    f0, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        cfg.fmin,
        cfg.fmax,
        model="full",
        return_periodicity=True,
        batch_size=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = f0[0].numpy().astype(np.float64)
    conf = periodicity[0].numpy().astype(np.float64)
    voiced = conf >= 0.5
    return np.where(voiced, f0, np.nan), voiced, conf


def hz_to_cents(f0: np.ndarray) -> np.ndarray:
    """Cents relative to A4 = 440 Hz (NaN stays NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1200.0 * np.log2(f0 / 440.0)
