"""Trimming leading/trailing silence and deciding whether the user sang only part of the song."""

from __future__ import annotations

import numpy as np

from app.models import SING, AudioFeatures


def active_range(feats: AudioFeatures, margin_s: float) -> tuple[int, int]:
    """[start, end) frame range spanning all singing, padded by `margin_s`."""
    sing = np.flatnonzero(feats.labels == SING)
    if sing.size == 0:
        return 0, feats.n_frames
    margin = int(round(margin_s / feats.hop))
    return max(0, int(sing[0]) - margin), min(feats.n_frames, int(sing[-1]) + 1 + margin)


def is_partial(user_len: int, ref_len: int, ratio: float = 0.75) -> bool:
    """True when the user's active part is much shorter than the reference (subsequence match)."""
    return user_len < ratio * ref_len
