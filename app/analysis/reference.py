"""Reference analysis -> ReferenceMap (with optional on-disk cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.analysis.analyzer import analyze_signal
from app.analysis.cache import cache_key, load_features, save_features
from app.config import Config
from app.models import AudioFeatures, ReferenceMap


def to_reference_map(features: AudioFeatures) -> ReferenceMap:
    return ReferenceMap(
        duration=features.duration,
        f0=features.f0,
        timestamps=features.timestamps,
        phrases=features.phrases,
        features=features,
    )


def analyze_reference(
    y: np.ndarray,
    cfg: Config,
    source: str | Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[ReferenceMap, bool]:
    """Analyse the reference vocal. Returns (map, cache_hit)."""
    key = None
    if cache_dir is not None and source is not None:
        key = cache_key(source, cfg.analysis_hash())
        cached = load_features(cache_dir, key)
        if cached is not None:
            return to_reference_map(cached), True

    features = analyze_signal(y, cfg)
    if key is not None:
        save_features(features, cache_dir, key)
    return to_reference_map(features), False
