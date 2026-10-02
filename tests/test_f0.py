import numpy as np
from synth import REFERENCE_MELODY, SR, synthesize

from app.analysis.analyzer import analyze_signal
from app.audio.resample import resample


def _analyze(y, cfg):
    return analyze_signal(resample(y, SR, cfg.audio.analysis_sample_rate), cfg)


def test_f0_tracks_known_melody(cfg):
    y, notes = synthesize(REFERENCE_MELODY, vibrato_cents=0)
    feats = _analyze(y, cfg)
    errors = []
    for n in notes:
        s, e = feats.frame_of(n.start + 0.08), feats.frame_of(n.end - 0.08)
        est = feats.midi[s:e]
        est = est[np.isfinite(est)]
        assert est.size > 0.7 * (e - s), f"note at {n.start:.2f}s mostly unvoiced"
        errors.append(np.median(np.abs(est - n.midi)) * 100)
    assert np.median(errors) < 20, errors


def test_phrases_notes_and_breaths(cfg):
    y, _ = synthesize(REFERENCE_MELODY)
    feats = _analyze(y, cfg)
    assert len(feats.phrases) == 3
    expected = [(0.5, 2.4), (3.0, 5.0), (5.5, 7.6)]
    for phrase, (s, e) in zip(feats.phrases, expected):
        assert abs(phrase.start - s) < 0.08
        assert abs(phrase.end - e) < 0.12
    n_notes = sum(len(p.notes) for p in feats.phrases)
    assert 12 <= n_notes <= 16, n_notes
    from app.models import BREATH

    assert (feats.labels == BREATH).sum() > 0
