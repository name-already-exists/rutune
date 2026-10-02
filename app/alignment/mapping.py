"""Turn an alignment into per-user-frame reference targets."""

from __future__ import annotations

import numpy as np

from app.analysis.f0 import hz_to_cents
from app.models import SING, AlignmentMap, AudioFeatures


def reference_cents_on_user(
    alignment: AlignmentMap, user: AudioFeatures, ref: AudioFeatures
) -> np.ndarray:
    """Reference pitch (cents re A4) for every user frame; NaN where the reference is unvoiced.

    Where the path maps one user frame to several reference frames (user faster than
    reference), the median of the voiced reference frames is used.
    """
    ref_cents = hz_to_cents(ref.f0)
    ref_cents[ref.labels != SING] = np.nan
    out = np.full(user.n_frames, np.nan)

    pu, pr = alignment.user_indices, alignment.reference_indices
    order = np.argsort(pu, kind="stable")
    pu, pr = pu[order], pr[order]
    uniq, start = np.unique(pu, return_index=True)
    bounds = np.append(start, len(pu))
    for i, u in enumerate(uniq):
        vals = ref_cents[pr[bounds[i] : bounds[i + 1]]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[u] = np.median(vals)

    missing = ~np.isfinite(out)
    nearest = np.clip(np.round(alignment.ref_position).astype(int), 0, ref.n_frames - 1)
    out[missing] = ref_cents[nearest[missing]]
    return out


def detect_octave(
    user_cents: np.ndarray, ref_cents: np.ndarray, enabled: bool = True
) -> tuple[float, float]:
    """Return (octave shift applied to the reference target, residual median offset in cents).

    A user singing a whole octave below/above the reference is not "wrong" (design §28.3);
    the target is moved by whole octaves to the user's register.
    """
    both = np.isfinite(user_cents) & np.isfinite(ref_cents)
    if both.sum() < 10:
        return 0.0, 0.0
    median = float(np.median(user_cents[both] - ref_cents[both]))
    shift = 1200.0 * round(median / 1200.0) if enabled else 0.0
    return shift, median - shift
