"""Input sanity checks performed before any heavy processing."""

from __future__ import annotations

import numpy as np

from app.models import Flag


class InputError(ValueError):
    pass


MIN_DURATION_S = 3.0
MIN_SAMPLE_RATE = 16000


def validate_input(y: np.ndarray, sr: int, orig_sr: int, name: str) -> list[Flag]:
    """Raise InputError for unusable audio; return warning flags for usable-but-suspect audio."""
    duration = len(y) / sr
    if duration < MIN_DURATION_S:
        raise InputError(f"{name}: audio is too short ({duration:.1f}s < {MIN_DURATION_S}s)")
    if orig_sr < MIN_SAMPLE_RATE:
        raise InputError(f"{name}: sample rate {orig_sr} Hz is below {MIN_SAMPLE_RATE} Hz")
    if float(np.max(np.abs(y))) < 1e-4:
        raise InputError(f"{name}: audio is silent")

    flags: list[Flag] = []
    clipped = np.abs(y) >= 0.999
    if clipped.mean() > 1e-4:
        flags.append(Flag("INPUT_CLIPPING", detail=f"{name}: {clipped.sum()} clipped samples"))
    return flags


def require_voice(n_sing_frames: int, hop: float, name: str) -> None:
    if n_sing_frames * hop < 1.0:
        raise InputError(f"{name}: no singing voice detected")
