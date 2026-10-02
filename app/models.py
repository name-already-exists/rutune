"""Core data model shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Frame labels produced by voice activity detection.
SILENCE, SING, UNVOICED, BREATH = 0, 1, 2, 3
LABEL_NAMES = {SILENCE: "silence", SING: "sing", UNVOICED: "unvoiced", BREATH: "breath"}


@dataclass
class Note:
    start: float
    end: float
    f0: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Phrase:
    start: float
    end: float
    confidence: float
    notes: list[Note] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AudioFeatures:
    """Frame-level analysis on a common hop grid (frame i is centred at i * hop seconds)."""

    sample_rate: int
    duration: float
    hop: float
    timestamps: np.ndarray
    f0: np.ndarray  # Hz, NaN where unvoiced
    f0_confidence: np.ndarray
    voiced: np.ndarray  # bool
    energy: np.ndarray  # RMS in dB relative to the loudest frame
    centroid: np.ndarray  # spectral centroid in Hz
    chroma: np.ndarray  # (12, n_frames)
    onsets: np.ndarray  # onset times in seconds
    labels: np.ndarray  # per-frame SILENCE/SING/UNVOICED/BREATH
    phrases: list[Phrase] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def midi(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return 69.0 + 12.0 * np.log2(self.f0 / 440.0)

    def frame_of(self, t: float) -> int:
        return int(np.clip(round(t / self.hop), 0, self.n_frames - 1))


@dataclass
class ReferenceMap:
    duration: float
    f0: np.ndarray
    timestamps: np.ndarray
    phrases: list[Phrase]
    features: AudioFeatures | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "phrases": [
                {
                    "start": round(p.start, 3),
                    "end": round(p.end, 3),
                    "confidence": round(p.confidence, 3),
                    "notes": [
                        {"start": round(n.start, 3), "end": round(n.end, 3), "f0": round(n.f0, 2)}
                        for n in p.notes
                    ],
                }
                for p in self.phrases
            ],
        }


@dataclass
class AlignmentMap:
    user_indices: np.ndarray  # fine warp path, user frame indices
    reference_indices: np.ndarray  # fine warp path, reference frame indices
    distance: float  # normalised path cost / mean pairwise cost (lower is better)
    ref_position: np.ndarray  # per user frame: (fractional) reference frame index
    offset: float  # reference time minus user time at the start of the user's singing
    user_range: tuple[int, int]  # active user frame range [start, end)
    failed: bool = False


@dataclass
class Flag:
    code: str
    start: float | None = None
    end: float | None = None
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class ProcessReport:
    reference: str = ""
    input: str = ""
    output: str = ""
    backend: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    pitch: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["flags"] = [f.to_json() for f in self.flags]
        return data
