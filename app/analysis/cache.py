"""On-disk cache for AudioFeatures, keyed by file content hash + analysis settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.models import AudioFeatures, Note, Phrase

_ARRAYS = (
    "timestamps", "f0", "f0_confidence", "voiced", "energy", "centroid", "chroma", "onsets",
    "labels",
)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(path: str | Path, config_hash: str) -> str:
    return f"{file_hash(path)}_{config_hash}"


def save_features(features: AudioFeatures, cache_dir: Path, key: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{key}.npz", **{k: getattr(features, k) for k in _ARRAYS})
    meta = {
        "sample_rate": features.sample_rate,
        "duration": features.duration,
        "hop": features.hop,
        "phrases": [asdict(p) for p in features.phrases],
    }
    (cache_dir / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")


def load_features(cache_dir: Path, key: str) -> AudioFeatures | None:
    npz, meta_path = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
    if not (npz.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz) as data:
        arrays = {k: data[k] for k in _ARRAYS}
    phrases = [
        Phrase(p["start"], p["end"], p["confidence"], [Note(**n) for n in p["notes"]])
        for p in meta["phrases"]
    ]
    return AudioFeatures(
        sample_rate=meta["sample_rate"],
        duration=meta["duration"],
        hop=meta["hop"],
        phrases=phrases,
        **arrays,
    )
