import numpy as np
from synth import (
    OTHER_MELODY,
    REFERENCE_MELODY,
    SR,
    detune_notes,
    stretch,
    synthesize,
    transpose,
)

from app.alignment.dtw import align
from app.analysis.analyzer import analyze_signal
from app.audio.resample import resample


def _analyze(y, cfg):
    return analyze_signal(resample(y, SR, cfg.audio.analysis_sample_rate), cfg)


def _note_center_error(user_notes, ref_notes, alignment, hop):
    errs = []
    for un, rn in zip(user_notes, ref_notes):
        uc = (un.start + un.end) / 2
        mapped = alignment.ref_position[int(round(uc / hop))] * hop
        errs.append(abs(mapped - (rn.start + rn.end) / 2))
    return np.array(errs)


def test_slower_user_maps_onto_reference_notes(cfg):
    ref_y, ref_notes = synthesize(REFERENCE_MELODY)
    user_y, user_notes = synthesize(stretch(REFERENCE_MELODY, 1.15, rest_factor=0.6), seed=1)
    ref, user = _analyze(ref_y, cfg), _analyze(user_y, cfg)
    alignment = align(user, ref, cfg.alignment, cfg.vad.energy_db)
    assert not alignment.failed
    errs = _note_center_error(user_notes, ref_notes, alignment, user.hop)
    assert np.median(errs) < 0.05, errs
    assert errs.max() < 0.15, errs
    assert np.all(np.diff(alignment.ref_position) >= 0)


def test_detuned_notes_stay_on_their_reference_notes(cfg):
    # ~55 cents off falls into the neighbouring chroma bin; the pitch term must hold the path.
    ref_y, ref_notes = synthesize(REFERENCE_MELODY)
    user_y, user_notes = synthesize(
        stretch(detune_notes(REFERENCE_MELODY, [55, -55, 60, -45]), 1.08, 0.7), seed=12
    )
    ref, user = _analyze(ref_y, cfg), _analyze(user_y, cfg)
    alignment = align(user, ref, cfg.alignment, cfg.vad.energy_db)
    errs = _note_center_error(user_notes, ref_notes, alignment, user.hop)
    assert errs.max() < 0.12, errs


def test_late_start_offset_and_octave_invariance(cfg):
    ref_y, ref_notes = synthesize(REFERENCE_MELODY)
    late = [(None, 2.0)] + transpose(REFERENCE_MELODY, -12)
    user_y, user_notes = synthesize(late, seed=2)
    ref, user = _analyze(ref_y, cfg), _analyze(user_y, cfg)
    alignment = align(user, ref, cfg.alignment, cfg.vad.energy_db)
    assert not alignment.failed
    assert abs(alignment.offset - (-2.0)) < 0.1, alignment.offset
    errs = _note_center_error(user_notes, ref_notes, alignment, user.hop)
    assert np.median(errs) < 0.05, errs


def test_different_song_is_rejected(cfg):
    ref_y, _ = synthesize(REFERENCE_MELODY)
    user_y, _ = synthesize(OTHER_MELODY, seed=3)
    ref, user = _analyze(ref_y, cfg), _analyze(user_y, cfg)
    same_y, _ = synthesize(REFERENCE_MELODY, seed=4)
    same = _analyze(same_y, cfg)

    bad = align(user, ref, cfg.alignment, cfg.vad.energy_db)
    good = align(same, ref, cfg.alignment, cfg.vad.energy_db)
    assert good.distance < bad.distance
    assert not good.failed
    assert bad.failed, bad.distance
