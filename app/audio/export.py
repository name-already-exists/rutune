"""Writing processed audio to disk."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def export_audio(y: np.ndarray, sr: int, path: str | Path) -> Path:
    """Write WAV (24-bit PCM) or, via ffmpeg, MP3 depending on the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    suffix = path.suffix.lower()
    if suffix in ("", ".wav"):
        sf.write(str(path.with_suffix(".wav")), y, sr, subtype="PCM_24")
        return path.with_suffix(".wav")
    if suffix in (".flac", ".ogg"):
        sf.write(str(path), y, sr)
        return path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"Exporting {suffix} requires ffmpeg on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        sf.write(str(wav), y, sr, subtype="PCM_24")
        cmd = [ffmpeg, "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(path)]
        subprocess.run(cmd, check=True)
    return path
