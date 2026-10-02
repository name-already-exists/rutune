"""Rubber Band backend (comparison only): constant shift per note, formant-preserving,
crossfaded back into the original signal. Requires the `rubberband` CLI (v3) on PATH.

Note: Rubber Band is GPL-licensed; commercial use needs a commercial licence.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import Config
from app.models import AudioFeatures

PAD_S = 0.05
FADE_S = 0.01
MIN_SHIFT_SEMITONES = 0.05


class RubberbandSegmentShifter:
    name = "rubberband"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.exe = shutil.which("rubberband") or shutil.which("rubberband-r3")
        if self.exe is None:
            raise RuntimeError("Backend 'rubberband' requires the rubberband CLI on PATH")

    def _shift(self, seg: np.ndarray, sr: int, semitones: float, tmp: Path) -> np.ndarray:
        src, dst = tmp / "in.wav", tmp / "out.wav"
        sf.write(str(src), seg, sr, subtype="FLOAT")
        cmd = [self.exe, "-q", "--fine", "-F", "-p", f"{semitones:.4f}", str(src), str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        out, _ = sf.read(str(dst), dtype="float32")
        out = out.mean(axis=1) if out.ndim > 1 else out
        if len(out) < len(seg):
            out = np.pad(out, (0, len(seg) - len(out)))
        return out[: len(seg)]

    def apply(
        self,
        y: np.ndarray,
        sr: int,
        correction_cents: np.ndarray,
        hop: float,
        user: AudioFeatures,
    ) -> np.ndarray:
        out = y.astype(np.float32).copy()
        pad, fade = int(PAD_S * sr), max(1, int(FADE_S * sr))
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for phrase in user.phrases:
                for note in phrase.notes:
                    f0s, f0e = user.frame_of(note.start), user.frame_of(note.end)
                    semis = float(np.median(correction_cents[f0s:max(f0e, f0s + 1)])) / 100.0
                    if abs(semis) < MIN_SHIFT_SEMITONES:
                        continue
                    s, e = int(note.start * sr), int(note.end * sr)
                    a, b = max(0, s - pad), min(len(y), e + pad)
                    shifted = self._shift(y[a:b], sr, semis, tmp)[s - a : e - a]
                    n = len(shifted)
                    if n <= 2 * fade:
                        continue
                    w = np.ones(n, dtype=np.float32)
                    w[:fade], w[-fade:] = ramp, ramp[::-1]
                    out[s:e] = out[s:e] * (1.0 - w) + shifted * w
        return out
