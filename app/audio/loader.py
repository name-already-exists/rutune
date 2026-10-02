"""Audio decoding: any common format -> mono float32 at the requested sample rate."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio.resample import resample


class AudioLoadError(RuntimeError):
    pass


def _decode_ffmpeg(path: Path, sr: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError(
            f"Cannot decode '{path.name}' with libsndfile and ffmpeg is not on PATH."
        )
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioLoadError(proc.stderr.decode(errors="replace").strip())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str | Path, sr: int) -> tuple[np.ndarray, int]:
    """Return (mono signal, original sample rate). The signal is resampled to `sr`."""
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")
    try:
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        return _decode_ffmpeg(path, sr), sr
    mono = data.mean(axis=1)
    return resample(mono, orig_sr, sr), orig_sr
