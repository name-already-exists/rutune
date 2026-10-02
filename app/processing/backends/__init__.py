"""Pitch/time resynthesis backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.config import Config
from app.models import AudioFeatures


class PitchShifter(Protocol):
    name: str

    def apply(
        self,
        y: np.ndarray,
        sr: int,
        correction_cents: np.ndarray,
        hop: float,
        user: AudioFeatures,
    ) -> np.ndarray:
        """Return `y` with its pitch moved by `correction_cents` (one value per analysis frame)."""
        ...


def get_backend(cfg: Config) -> PitchShifter:
    if cfg.backend == "world":
        from app.processing.backends.world import WorldShifter

        return WorldShifter(cfg)
    if cfg.backend == "rubberband":
        from app.processing.backends.rubberband import RubberbandSegmentShifter

        return RubberbandSegmentShifter(cfg)
    raise ValueError(f"Unknown backend '{cfg.backend}' (use world | rubberband)")
