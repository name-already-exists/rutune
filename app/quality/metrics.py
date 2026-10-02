"""Performance metrics against the reference (design §30)."""

from __future__ import annotations

import numpy as np


def pitch_error(cents: np.ndarray, target_cents: np.ndarray) -> dict[str, float]:
    """Cents error on frames where both curves are defined."""
    both = np.isfinite(cents) & np.isfinite(target_cents)
    if not both.any():
        return {"frames": 0, "mae_cents": float("nan"), "median_cents": float("nan"),
                "within_50c": float("nan")}
    err = np.abs(cents[both] - target_cents[both])
    return {
        "frames": int(both.sum()),
        "mae_cents": round(float(err.mean()), 1),
        "median_cents": round(float(np.median(err)), 1),
        "within_50c": round(float((err <= 50).mean()), 3),
    }
