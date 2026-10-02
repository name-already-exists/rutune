"""WORLD vocoder backend: change F0, keep the spectral envelope (timbre/formants) untouched."""

from __future__ import annotations

import numpy as np
import pyworld as pw

from app.config import Config
from app.models import AudioFeatures


class WorldShifter:
    name = "world"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_period = self.cfg.audio.hop_ms
        fmin, fmax = self.cfg.f0.fmin, self.cfg.f0.fmax
        if self.cfg.world.f0_method == "dio":
            f0, t = pw.dio(x, sr, f0_floor=fmin, f0_ceil=fmax, frame_period=frame_period)
            f0 = pw.stonemask(x, f0, t, sr)
        else:
            f0, t = pw.harvest(x, sr, f0_floor=fmin, f0_ceil=fmax, frame_period=frame_period)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)
        return f0, t, sp, ap

    def apply(
        self,
        y: np.ndarray,
        sr: int,
        correction_cents: np.ndarray,
        hop: float,
        user: AudioFeatures,
    ) -> np.ndarray:
        x = np.ascontiguousarray(y, dtype=np.float64)
        f0, t, sp, ap = self.analyze(x, sr)
        grid = np.arange(len(correction_cents)) * hop
        corr = np.interp(t, grid, correction_cents)
        new_f0 = np.where(f0 > 0, f0 * 2.0 ** (corr / 1200.0), 0.0)
        out = pw.synthesize(new_f0, sp, ap, sr, frame_period=self.cfg.audio.hop_ms)
        if len(out) < len(y):
            out = np.pad(out, (0, len(y) - len(out)))
        return out[: len(y)].astype(np.float32)
