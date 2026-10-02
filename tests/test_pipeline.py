import json

import soundfile as sf
from synth import OTHER_MELODY, REFERENCE_MELODY, SR, detune_notes, stretch, synthesize
from typer.testing import CliRunner

from app.cli import cli


def _write(path, melody, seed):
    y, _ = synthesize(melody, seed=seed)
    sf.write(str(path), y, SR)
    return path


def test_cli_process_end_to_end(tmp_path):
    ref = _write(tmp_path / "ref.wav", REFERENCE_MELODY, 0)
    user_melody = stretch(detune_notes(REFERENCE_MELODY, [70, -50]), 1.1)
    user = _write(tmp_path / "user.wav", [(None, 1.0)] + user_melody, 1)
    out = tmp_path / "out" / "result.wav"
    debug = tmp_path / "debug"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("audio:\n  sample_rate: 22050\nworld:\n  f0_method: dio\n", encoding="utf-8")

    args = ["process", "-r", str(ref), "-i", str(user), "-o", str(out), "-s", "close",
            "-c", str(cfg), "--debug-dir", str(debug), "--no-cache"]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 0, res.output

    y, sr = sf.read(str(out))
    assert sr == 22050 and len(y) > 0
    report = json.loads((out.parent / "result.report.json").read_text(encoding="utf-8"))
    m = report["metrics"]
    assert m["pitch_after"]["mae_cents"] < m["pitch_before"]["mae_cents"]
    assert report["settings"]["pitch"]["correction_strength"] == 0.85
    for name in ("pitch_correction.png", "alignment.png", "user_analysis.png",
                 "reference_map.json", "curves.npz"):
        assert (debug / name).is_file(), name


def test_cli_rejects_mismatched_song(tmp_path):
    ref = _write(tmp_path / "ref.wav", REFERENCE_MELODY, 0)
    user = _write(tmp_path / "user.wav", OTHER_MELODY, 3)
    out = tmp_path / "result.wav"
    res = CliRunner().invoke(cli, ["process", "-r", str(ref), "-i", str(user), "-o", str(out),
                                   "--no-cache"])
    assert res.exit_code == 2
    report = json.loads((tmp_path / "result.report.json").read_text(encoding="utf-8"))
    assert any(f["code"] == "ALIGNMENT_FAILED" for f in report["flags"])


def test_cli_rejects_too_short_input(tmp_path):
    ref = _write(tmp_path / "ref.wav", REFERENCE_MELODY, 0)
    short = _write(tmp_path / "short.wav", [(60, 1.0)], 1)
    res = CliRunner().invoke(cli, ["process", "-r", str(ref), "-i", str(short), "-o",
                                   str(tmp_path / "o.wav"), "--no-cache"])
    assert res.exit_code == 2
    assert "too short" in res.output
