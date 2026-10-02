"""Voice activity detection: per-frame silence / singing / unvoiced / breath labels."""

from __future__ import annotations

import numpy as np

from app.config import VadConfig
from app.models import BREATH, SILENCE, SING, UNVOICED


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def vad_labels(
    energy_db: np.ndarray, voiced: np.ndarray, confidence: np.ndarray, cfg: VadConfig
) -> np.ndarray:
    """Singing = energy above threshold AND confident F0 (design §6)."""
    labels = np.full(len(energy_db), SILENCE, dtype=np.int8)
    active = energy_db >= cfg.energy_db
    sing = active & voiced & (confidence >= cfg.f0_conf)
    labels[active] = UNVOICED
    labels[sing] = SING
    return labels


def mark_breaths(
    labels: np.ndarray,
    centroid: np.ndarray,
    in_phrase: np.ndarray,
    hop: float,
    cfg: VadConfig,
) -> np.ndarray:
    """Relabel long, bright, unvoiced runs outside phrases as BREATH.

    Short unvoiced runs inside phrases are consonants and stay UNVOICED.
    """
    labels = labels.copy()
    min_frames = max(1, int(round(cfg.breath_min_ms / 1000.0 / hop)))
    for start, end in find_runs((labels == UNVOICED) & ~in_phrase):
        if end - start < min_frames:
            continue
        if float(np.median(centroid[start:end])) >= cfg.breath_centroid_hz:
            labels[start:end] = BREATH
    return labels
