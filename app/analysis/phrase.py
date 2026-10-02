"""Phrase and note segmentation from frame-level features."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from app.audio.silence import find_runs
from app.config import NoteConfig, VadConfig
from app.models import SING, Note, Phrase


def detect_phrase_frames(labels: np.ndarray, hop: float, cfg: VadConfig) -> list[tuple[int, int]]:
    """Group singing frames into phrases: gaps shorter than min_pause are bridged."""
    min_gap = int(round(cfg.min_pause_ms / 1000.0 / hop))
    min_len = int(round(cfg.min_phrase_ms / 1000.0 / hop))
    merged: list[list[int]] = []
    for start, end in find_runs(labels == SING):
        if merged and start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_len]


def _midi(f0: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 69.0 + 12.0 * np.log2(f0 / 440.0)


def segment_notes(
    f0: np.ndarray,
    voiced: np.ndarray,
    onsets: np.ndarray,
    start: int,
    end: int,
    hop: float,
    cfg: NoteConfig,
) -> list[Note]:
    """Split one phrase into notes at voicing gaps, sustained pitch jumps and onsets."""
    min_len = max(2, int(round(cfg.min_note_ms / 1000.0 / hop)))
    midi = _midi(f0[start:end])
    is_voiced = voiced[start:end] & np.isfinite(midi)
    onset_frames = {int(round(t / hop)) - start for t in onsets}

    bounds: list[tuple[int, int]] = []
    for vs, ve in find_runs(is_voiced):
        seg = median_filter(midi[vs:ve], size=5, mode="nearest") if ve - vs >= 5 else midi[vs:ve]
        note_start, ref, jump = vs, seg[0], 0
        for i in range(1, ve - vs):
            idx = vs + i
            if abs(seg[i] - ref) > cfg.split_semitones:
                jump += 1
                if jump >= 3:
                    cut = idx - 2
                    if cut - note_start >= min_len:
                        bounds.append((note_start, cut))
                        note_start = cut
                    ref, jump = seg[i], 0
                    continue
            else:
                jump = 0
            if idx in onset_frames and idx - note_start >= min_len and ve - idx >= min_len:
                bounds.append((note_start, idx))
                note_start, ref = idx, seg[i]
                continue
            # Track slow drift without following vibrato.
            ref = 0.95 * ref + 0.05 * seg[i] if abs(seg[i] - ref) <= cfg.split_semitones else ref
        bounds.append((note_start, ve))

    notes: list[Note] = []
    for s, e in bounds:
        if e - s < min_len and notes:
            prev = notes[-1]
            notes[-1] = Note(prev.start, (start + e) * hop, prev.f0)
            continue
        m = np.nanmedian(midi[s:e])
        notes.append(Note((start + s) * hop, (start + e) * hop, float(440.0 * 2 ** ((m - 69) / 12))))
    return notes


def build_phrases(
    labels: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
    confidence: np.ndarray,
    onsets: np.ndarray,
    hop: float,
    vad_cfg: VadConfig,
    note_cfg: NoteConfig,
) -> tuple[list[Phrase], list[tuple[int, int]]]:
    frames = detect_phrase_frames(labels, hop, vad_cfg)
    phrases = []
    for s, e in frames:
        sing = labels[s:e] == SING
        conf = float(sing.mean() * np.mean(confidence[s:e][sing])) if sing.any() else 0.0
        notes = segment_notes(f0, voiced, onsets, s, e, hop, note_cfg)
        phrases.append(Phrase(s * hop, e * hop, conf, notes))
    return phrases, frames
