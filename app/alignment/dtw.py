"""Two-stage DTW: coarse whole-song alignment, then fine per-phrase refinement.

Full-resolution DTW over a whole song is O(N*M) memory (a 4-minute song at 10 ms is
~24k x 24k cells), so we align at a coarse hop first and refine each user phrase inside a
small reference window.
"""

from __future__ import annotations

import math

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from app.alignment.offset import active_range, is_partial
from app.config import AlignmentConfig
from app.models import SING, AlignmentMap, AudioFeatures

PITCH_SCALE_SEMITONES = 3.0  # 1 semitone of relative pitch = 0.33 feature units


def feature_matrix(feats: AudioFeatures, cfg: AlignmentConfig, vad_energy_db: float) -> np.ndarray:
    """Per-frame alignment features (d x n): chroma + relative pitch + energy + voicing.

    Pitch is taken relative to the singer's median so octave/key offsets do not hurt matching.
    """
    active = feats.energy >= vad_energy_db
    sing = feats.labels == SING

    chroma = feats.chroma * active[None, :]

    midi = feats.midi
    valid = np.isfinite(midi) & sing
    rel = np.zeros(feats.n_frames)
    if valid.sum() >= 2:
        idx = np.arange(feats.n_frames)
        rel = np.interp(idx, idx[valid], midi[valid]) - np.median(midi[valid])
    # Chroma snaps a note sung ~50 cents sharp into the neighbouring bin, so a continuous
    # pitch term is needed to keep detuned notes matched to the right reference note.
    rel = np.clip(rel / PITCH_SCALE_SEMITONES, -4.0, 4.0)

    act_energy = feats.energy[active]
    center = float(np.median(act_energy)) if act_energy.size else 0.0
    energy = np.clip((feats.energy - center) / 20.0, -3.0, 1.0)

    return np.vstack(
        [
            cfg.weight_chroma * chroma,
            cfg.weight_pitch * rel[None, :],
            cfg.weight_energy * energy[None, :],
            cfg.weight_voiced * sing[None, :].astype(float),
        ]
    )


def _pool(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    n = x.shape[1] // k * k
    pooled = x[:, :n].reshape(x.shape[0], -1, k).mean(axis=2)
    if n < x.shape[1]:
        pooled = np.hstack([pooled, x[:, n:].mean(axis=1, keepdims=True)])
    return pooled


_STEPS = np.array([[1, 1], [0, 1], [1, 0]])


def _dtw(
    x: np.ndarray, y: np.ndarray, subseq: bool, step_penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (path as (k, 2) array in forward order [x_idx, y_idx], cost matrix).

    Non-diagonal steps pay `step_penalty`; without it, a sustained note can be mapped onto a
    single frame of the other recording at no extra cost (the path "sticks").
    """
    cost = cdist(x.T, y.T, metric="euclidean")
    _, wp = librosa.sequence.dtw(
        C=cost,
        step_sizes_sigma=_STEPS,
        weights_add=np.array([0.0, step_penalty, step_penalty]),
        weights_mul=np.ones(3),
        subseq=subseq,
        backtrack=True,
    )
    return wp[::-1].copy(), cost


def align(
    user: AudioFeatures,
    ref: AudioFeatures,
    cfg: AlignmentConfig,
    vad_energy_db: float,
) -> AlignmentMap:
    fu = feature_matrix(user, cfg, vad_energy_db)
    fr = feature_matrix(ref, cfg, vad_energy_db)
    us, ue = active_range(user, cfg.trim_margin_s)
    rs, re = active_range(ref, cfg.trim_margin_s)
    subseq = is_partial(ue - us, re - rs)

    # ---- Stage 1: coarse alignment of the active regions --------------------------------
    k = max(1, int(round(cfg.coarse_hop_ms / 1000.0 / user.hop)))
    cells = (ue - us) * (re - rs)
    k = max(k, math.ceil(math.sqrt(cells / cfg.max_coarse_cells)))
    xc, yc = _pool(fu[:, us:ue], k), _pool(fr[:, rs:re], k)
    coarse_path, coarse_cost = _dtw(xc, yc, subseq=subseq, step_penalty=cfg.step_penalty)
    path_cost = float(coarse_cost[coarse_path[:, 0], coarse_path[:, 1]].mean())
    ratio = path_cost / max(float(coarse_cost.mean()), 1e-9)

    # Coarse path in fine frame units (centre of each pooled block).
    cu = us + coarse_path[:, 0] * k + k // 2
    cr = rs + coarse_path[:, 1] * k + k // 2

    # ---- Stage 2: fine refinement per user phrase chunk ---------------------------------
    pad = int(round(cfg.refine_pad_s / user.hop))
    max_chunk = int(round(cfg.refine_max_chunk_s / user.hop))
    fine_u: list[np.ndarray] = []
    fine_r: list[np.ndarray] = []
    for p in user.phrases:
        ps, pe = user.frame_of(p.start), user.frame_of(p.end)
        for cs in range(ps, pe, max_chunk):
            ce = min(pe, cs + max_chunk)
            if ce - cs < 3:
                continue
            sel = (cu >= cs - k) & (cu <= ce + k)
            if not sel.any():
                continue
            w0 = max(0, int(cr[sel].min()) - pad)
            w1 = min(ref.n_frames, int(cr[sel].max()) + pad)
            if w1 - w0 < 3:
                continue
            path, _ = _dtw(fu[:, cs:ce], fr[:, w0:w1], subseq=True, step_penalty=cfg.step_penalty)
            fine_u.append(path[:, 0] + cs)
            fine_r.append(path[:, 1] + w0)

    if fine_u:
        pu, pr = np.concatenate(fine_u), np.concatenate(fine_r)
    else:
        pu, pr = cu, cr

    ref_position = _position_per_frame(pu, pr, user.n_frames, ref.n_frames)
    return AlignmentMap(
        user_indices=pu,
        reference_indices=pr,
        distance=ratio,
        ref_position=ref_position,
        offset=float((ref_position[us] - us) * user.hop),
        user_range=(us, ue),
        failed=ratio > cfg.max_cost_ratio,
    )


def _position_per_frame(pu: np.ndarray, pr: np.ndarray, n_user: int, n_ref: int) -> np.ndarray:
    """Collapse a (possibly many-to-many) path into a monotonic user->ref position curve."""
    order = np.argsort(pu, kind="stable")
    pu, pr = pu[order], pr[order]
    uniq, start = np.unique(pu, return_index=True)
    sums = np.add.reduceat(pr.astype(float), start)
    counts = np.diff(np.append(start, len(pu)))
    anchors = sums / counts

    frames = np.arange(n_user, dtype=float)
    pos = np.interp(frames, uniq, anchors)
    # Outside the path extrapolate with unit slope (constant offset) rather than holding.
    pos[frames < uniq[0]] = anchors[0] - (uniq[0] - frames[frames < uniq[0]])
    pos[frames > uniq[-1]] = anchors[-1] + (frames[frames > uniq[-1]] - uniq[-1])
    pos = np.maximum.accumulate(pos)
    return np.clip(pos, 0, n_ref - 1)
