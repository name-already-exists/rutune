"""Engine configuration: dataclasses, YAML loading and strength presets."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    analysis_sample_rate: int = 16000
    hop_ms: float = 10.0
    normalize_peak_db: float = -1.0


@dataclass
class F0Config:
    method: str = "pyin"  # pyin | crepe
    fmin: float = 65.0
    fmax: float = 1100.0
    frame_length: int = 1024  # at analysis_sample_rate


@dataclass
class VadConfig:
    energy_db: float = -45.0  # relative to the loudest frame
    f0_conf: float = 0.3  # pYIN voiced probability is often ~0.4 on low notes
    min_pause_ms: float = 250.0
    min_phrase_ms: float = 200.0
    breath_min_ms: float = 120.0
    breath_centroid_hz: float = 1500.0


@dataclass
class NoteConfig:
    split_semitones: float = 0.8
    min_note_ms: float = 60.0


@dataclass
class AlignmentConfig:
    coarse_hop_ms: float = 50.0
    max_coarse_cells: int = 8_000_000
    refine_pad_s: float = 0.5
    refine_max_chunk_s: float = 10.0
    max_cost_ratio: float = 0.55  # synthetic: same song <= 0.28, other song >= 0.76
    step_penalty: float = 0.15  # extra cost of a non-diagonal DTW step
    trim_margin_s: float = 0.2
    weight_chroma: float = 0.5
    weight_pitch: float = 1.0
    weight_energy: float = 0.5
    weight_voiced: float = 0.5


@dataclass
class PitchConfig:
    correction_strength: float = 0.60
    max_shift_semitones: float = 4.0
    smoothing: float = 0.15  # seconds, Gaussian window applied to the correction curve
    octave_detect: bool = True
    key_mismatch_cents: float = 150.0


@dataclass
class TimingConfig:
    enabled: bool = True
    strength: float = 0.55
    min_ratio: float = 0.67
    max_ratio: float = 1.5


@dataclass
class PauseConfig:
    enabled: bool = True
    strength: float = 0.50
    keep_breath: bool = True


@dataclass
class DynamicsConfig:
    transfer: bool = True
    strength: float = 0.4
    max_gain_db: float = 6.0
    compression: bool = True
    limiter: bool = True
    target_lufs: float = -16.0


@dataclass
class WorldConfig:
    f0_method: str = "harvest"  # harvest | dio


@dataclass
class OutputConfig:
    format: str = "wav"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    f0: F0Config = field(default_factory=F0Config)
    vad: VadConfig = field(default_factory=VadConfig)
    notes: NoteConfig = field(default_factory=NoteConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    pause: PauseConfig = field(default_factory=PauseConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    backend: str = "world"  # world | rubberband
    world: WorldConfig = field(default_factory=WorldConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def analysis_hash(self) -> str:
        """Hash of every setting that influences analysis results (used as cache key)."""
        relevant = {k: self.to_dict()[k] for k in ("audio", "f0", "vad", "notes")}
        return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "natural": {
        "pitch": {"correction_strength": 0.35},
        "timing": {"strength": 0.30},
        "pause": {"strength": 0.25},
    },
    "balanced": {
        "pitch": {"correction_strength": 0.60},
        "timing": {"strength": 0.55},
        "pause": {"strength": 0.50},
    },
    "close": {
        "pitch": {"correction_strength": 0.85},
        "timing": {"strength": 0.80},
        "pause": {"strength": 0.75},
    },
}


def _merge(obj: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{key}' must be a mapping")
            _merge(current, value)
        else:
            setattr(obj, key, type(current)(value) if current is not None else value)


def load_config(path: str | Path | None = None, strength: str | None = None) -> Config:
    """Build a Config from defaults, an optional YAML file, then an optional preset."""
    cfg = Config()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(cfg, data)
    if strength is not None:
        if strength not in PRESETS:
            raise ValueError(f"Unknown strength preset '{strength}'. Use: {', '.join(PRESETS)}")
        _merge(cfg, PRESETS[strength])
    return cfg
