"""Output-side artifact checks (design §29)."""

from __future__ import annotations

import numpy as np

from app.models import Flag

CLIP_LEVEL = 0.999
SAFE_PEAK = 10 ** (-1.0 / 20.0)


def protect_clipping(y: np.ndarray) -> tuple[np.ndarray, list[Flag]]:
    """Scale down if resynthesis pushed the peak over full scale; flag it."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < CLIP_LEVEL:
        return y, []
    flag = Flag("CLIPPING", detail=f"peak {peak:.3f} > full scale, gain reduced to -1 dBFS")
    return (y * (SAFE_PEAK / peak)).astype(np.float32), [flag]
