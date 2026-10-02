import numpy as np
import pytest
from synth import REFERENCE_MELODY, SR, detune_notes, synthesize, transpose

from app.analysis.f0 import hz_to_cents
from app.analysis.reference import analyze_reference
from app.audio.resample import resample
from app.models import SING
from app.pipeline.processor import process_audio


@pytest.fixture(scope="module")
def reference_audio():
    y, notes = synthesize(REFERENCE_MELODY)
    return y, notes


def _ref_map(y, cfg):
    ref, _ = analyze_reference(resample(y, SR, cfg.audio.analysis_sample_rate), cfg)
    return ref


def test_detuned_user_moves_toward_reference(cfg, reference_audio):
    ref_y, _ = reference_audio
    cfg.pitch.correction_strength = 0.85
    detuned = detune_notes(REFERENCE_MELODY, [80, 60, -70, 90, 50])
    user_y, _ = synthesize(detuned, seed=5)
    result = process_audio(ref_y, user_y, cfg, _ref_map(ref_y, cfg))

    before = result.report.metrics["pitch_before"]["mae_cents"]
    after = result.report.metrics["pitch_after"]["mae_cents"]
    assert before > 50
    assert after < before * 0.5, (before, after)
    # WORLD resynthesis floor is ~5 dB; a formant-shifting pitch shift scores ~8.5 dB.
    assert result.report.metrics["voice"]["mcd_db"] < 7.0
    assert len(result.audio) == len(user_y)


def test_strength_scales_correction(cfg, reference_audio):
    ref_y, _ = reference_audio
    user_y, _ = synthesize(detune_notes(REFERENCE_MELODY, [100]), seed=6)
    ref_map = _ref_map(ref_y, cfg)
    out = {}
    for strength in (0.3, 0.85):
        cfg.pitch.correction_strength = strength
        res = process_audio(ref_y, user_y, cfg, ref_map)
        out[strength] = res.report.metrics["pitch_after"]["mae_cents"]
    assert out[0.85] < out[0.3]


def test_octave_lower_user_is_not_forced_up(cfg, reference_audio):
    ref_y, _ = reference_audio
    user_y, _ = synthesize(transpose(REFERENCE_MELODY, -12.3), seed=7)
    result = process_audio(ref_y, user_y, cfg, _ref_map(ref_y, cfg), evaluate=False)
    assert result.plan.octave_shift == -1200
    assert abs(result.plan.residual_offset + 30) < 15
    assert not any(f.code == "LARGE_PITCH_SHIFT" for f in result.report.flags)


def test_vibrato_is_preserved(cfg, reference_audio):
    ref_y, _ = reference_audio
    cfg.pitch.correction_strength = 0.85
    user_y, user_notes = synthesize(detune_notes(REFERENCE_MELODY, [60]), vibrato_cents=40,
                                    seed=8)
    result = process_audio(ref_y, user_y, cfg, _ref_map(ref_y, cfg))
    proc = result.processed_features
    cents = hz_to_cents(proc.f0)
    cents[proc.labels != SING] = np.nan
    longest = max(user_notes, key=lambda n: n.end - n.start)
    s, e = proc.frame_of(longest.start + 0.3), proc.frame_of(longest.end - 0.1)
    seg = cents[s:e]
    seg = seg[np.isfinite(seg)]
    assert seg.size > 20
    detrended = seg - np.polyval(np.polyfit(np.arange(seg.size), seg, 1), np.arange(seg.size))
    assert detrended.std() > 12, detrended.std()  # 40 cent sine -> ~28 cents std


def test_wrong_note_is_clamped_and_flagged(cfg, reference_audio):
    ref_y, _ = reference_audio
    wrong = detune_notes(REFERENCE_MELODY, [0, 0, 0, 0, 0, 0, 700, 0, 0, 0, 0, 0])
    user_y, _ = synthesize(wrong, seed=9)
    cfg.pitch.correction_strength = 1.0
    result = process_audio(ref_y, user_y, cfg, _ref_map(ref_y, cfg), evaluate=False)
    flags = [f for f in result.report.flags if f.code == "LARGE_PITCH_SHIFT"]
    assert flags, result.report.flags
    assert np.max(np.abs(result.plan.correction_cents)) <= cfg.pitch.max_shift_semitones * 100
