"""Reference-guided pitch correction curve (design §9-10, §28.2-28.3).

The correction is a smooth curve in cents added to the user's own F0. Smoothing is applied
to the *correction*, not to the F0, so the user's vibrato and micro-variation survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.analysis.f0 import hz_to_cents
from app.audio.silence import find_runs
from app.config import PitchConfig
from app.models import SING, AudioFeatures, Flag

MIN_FLAG_S = 0.1
LOW_CONFIDENCE = 0.45


@dataclass
class PitchPlan:
    user_cents: np.ndarray  # user F0 (NaN where not singing)
    target_cents: np.ndarray  # mapped reference + octave shift (NaN where undefined)
    correction_cents: np.ndarray  # smooth curve, defined on every user frame
    clamped: np.ndarray  # frames where the requested shift exceeded max_shift_semitones
    octave_shift: float
    residual_offset: float
    flags: list[Flag] = field(default_factory=list)

    def summary(self) -> dict:
        sing = np.isfinite(self.user_cents)
        corr = np.abs(self.correction_cents[sing]) if sing.any() else np.zeros(1)
        return {
            "octave_shift_cents": self.octave_shift,
            "residual_offset_cents": round(self.residual_offset, 1),
            "mean_abs_correction_cents": round(float(corr.mean()), 1),
            "max_abs_correction_cents": round(float(corr.max()), 1),
            "clamped_ratio": round(float(self.clamped[sing].mean()) if sing.any() else 0.0, 4),
        }


def plan_correction(
    user: AudioFeatures,
    ref_cents_on_user: np.ndarray,
    octave_shift: float,
    residual: float,
    cfg: PitchConfig,
) -> PitchPlan:
    hop = user.hop
    user_cents = hz_to_cents(user.f0)
    user_cents[user.labels != SING] = np.nan
    target = ref_cents_on_user + octave_shift

    error = target - user_cents
    defined = np.isfinite(error)
    limit = cfg.max_shift_semitones * 100.0
    requested = error * cfg.correction_strength
    clamped = defined & (np.abs(requested) > limit)
    raw = np.clip(np.where(defined, requested, np.nan), -limit, limit)

    # Singing runs with no reference pitch at all (e.g. the user holds a note past the
    # reference) get no correction rather than a value borrowed from a neighbouring note.
    for s, e in find_runs(np.isfinite(user_cents)):
        if not defined[s:e].any():
            raw[s:e] = 0.0

    idx = np.arange(len(raw))
    known = np.isfinite(raw)
    if known.any():
        filled = np.interp(idx, idx[known], raw[known])
    else:
        filled = np.zeros(len(raw))

    sigma = max(cfg.smoothing / 2.355 / hop, 0.0)  # smoothing is the Gaussian FWHM in seconds
    correction = gaussian_filter1d(filled, sigma) if sigma > 0 else filled
    correction = np.clip(correction, -limit, limit)

    flags: list[Flag] = []
    min_frames = int(round(MIN_FLAG_S / hop))
    for s, e in find_runs(clamped):
        if e - s >= min_frames:
            peak = float(np.nanmax(np.abs(error[s:e]))) / 100.0
            flags.append(
                Flag("LARGE_PITCH_SHIFT", s * hop, e * hop, f"needed {peak:.1f} st, clamped")
            )
    if abs(residual) > cfg.key_mismatch_cents:
        flags.append(
            Flag("KEY_MISMATCH", detail=f"user is {residual:+.0f} cents off the reference key")
        )
    sing_conf = user.f0_confidence[user.labels == SING]
    if sing_conf.size and float(np.mean(sing_conf)) < LOW_CONFIDENCE:
        flags.append(
            Flag("LOW_F0_CONFIDENCE", detail=f"mean user F0 confidence {np.mean(sing_conf):.2f}")
        )

    return PitchPlan(user_cents, target, correction, clamped, octave_shift, residual, flags)
