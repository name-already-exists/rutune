"""Synthetic 'singing' generator with known ground truth (pitch, timing, pauses).

A harmonic source shaped by fixed vowel-like formants, with vibrato, legato glides,
attack/release envelopes and optional breath noise in the rests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22050

# (midi or None for a rest, duration in seconds); phrases are separated by rests.
REFERENCE_MELODY: list[tuple[float | None, float]] = [
    (None, 0.5),
    (60, 0.4), (62, 0.4), (64, 0.4), (65, 0.7),
    (None, 0.6),
    (67, 0.5), (65, 0.35), (64, 0.35), (62, 0.8),
    (None, 0.5),
    (60, 0.4), (64, 0.4), (67, 0.4), (72, 0.9),
    (None, 0.5),
]

OTHER_MELODY: list[tuple[float | None, float]] = [
    (None, 0.3),
    (70, 1.2), (58, 0.2), (59, 0.2), (69, 1.0),
    (None, 0.2),
    (55, 0.25), (66, 0.25), (55, 0.25), (66, 0.25), (57, 1.5),
    (None, 0.2),
    (71, 0.3), (53, 1.1), (63, 0.6),
    (None, 0.3),
]

FORMANTS = [(800, 80), (1200, 90), (2500, 120), (3300, 150)]


@dataclass
class SynthNote:
    start: float
    end: float
    midi: float


def _envelope(freqs: np.ndarray) -> np.ndarray:
    amp = np.zeros_like(freqs)
    for f, bw in FORMANTS:
        amp += np.exp(-0.5 * ((freqs - f) / bw) ** 2)
    return (0.15 + amp) / np.maximum(freqs / 200.0, 1.0)


def stretch(melody, factor: float, rest_factor: float | None = None):
    rest_factor = factor if rest_factor is None else rest_factor
    return [(m, d * (rest_factor if m is None else factor)) for m, d in melody]


def transpose(melody, semitones: float):
    return [(None if m is None else m + semitones, d) for m, d in melody]


def synthesize(
    melody,
    sr: int = SR,
    vibrato_cents: float = 30.0,
    vibrato_hz: float = 5.5,
    breath: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, list[SynthNote]]:
    rng = np.random.default_rng(seed)
    total = sum(d for _, d in melody)
    n = int(total * sr)
    midi_track = np.full(n, np.nan)
    gain = np.zeros(n)
    noise = np.zeros(n)
    notes: list[SynthNote] = []

    t = 0.0
    for m, d in melody:
        s, e = int(t * sr), min(n, int((t + d) * sr))
        if m is None:
            if breath and d >= 0.4:
                b0 = s + int(0.1 * (e - s))
                b1 = b0 + int(0.25 * sr)
                env = np.hanning(b1 - b0)
                noise[b0:b1] += 0.02 * env * rng.standard_normal(b1 - b0)
        else:
            midi_track[s:e] = m
            local = np.arange(e - s) / sr
            vib_on = np.clip((local - 0.15) / 0.1, 0, 1)
            midi_track[s:e] += vib_on * vibrato_cents / 100 * np.sin(2 * np.pi * vibrato_hz * local)
            g = np.ones(e - s)
            a, r = int(0.03 * sr), int(0.05 * sr)
            g[:a] = np.linspace(0, 1, a)
            g[-r:] = np.linspace(1, 0, r)
            gain[s:e] = 0.5 * g
            notes.append(SynthNote(t, t + d, float(m)))
        t += d

    # Legato glide: smooth the pitch track inside voiced spans (~40 ms).
    voiced = np.isfinite(midi_track)
    filled = np.where(voiced, midi_track, 0.0)
    k = int(0.04 * sr)
    kernel = np.ones(k) / k
    smooth = np.convolve(filled, kernel, mode="same") / np.maximum(
        np.convolve(voiced.astype(float), kernel, mode="same"), 1e-6
    )
    midi_track = np.where(voiced, smooth, np.nan)

    f0 = np.where(voiced, 440.0 * 2 ** ((np.nan_to_num(midi_track, nan=69) - 69) / 12), 0.0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(n)
    max_h = int((sr / 2 * 0.9) / 100)
    for h in range(1, max_h + 1):
        fh = f0 * h
        ok = (fh > 0) & (fh < sr / 2 * 0.9)
        if not ok.any():
            break
        y += np.where(ok, _envelope(fh) * np.sin(h * phase), 0.0)
    y = y / max(np.max(np.abs(y)), 1e-9) * gain * 1.6
    hp = np.diff(noise, prepend=0.0)  # crude high-pass -> bright breath
    y += hp * 4 + 1e-4 * rng.standard_normal(n)
    y = y / np.max(np.abs(y)) * 0.8
    return y.astype(np.float32), notes


def detune_notes(melody, cents_by_note: list[float]):
    """Per-note pitch errors (cents) applied in order to the sung notes."""
    out, i = [], 0
    for m, d in melody:
        if m is None:
            out.append((m, d))
        else:
            out.append((m + cents_by_note[i % len(cents_by_note)] / 100.0, d))
            i += 1
    return out
