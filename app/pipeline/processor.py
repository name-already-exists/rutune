"""End-to-end processing: reference + user vocal -> processed user vocal + report.

`process_audio` is a pure function on arrays (reusable from an API/worker later);
`process_files` adds file I/O, caching, report and debug output.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.alignment.dtw import align
from app.alignment.mapping import detect_octave, reference_cents_on_user
from app.analysis.analyzer import analyze_signal
from app.analysis.f0 import hz_to_cents
from app.analysis.reference import analyze_reference
from app.audio.export import export_audio
from app.audio.loader import load_audio
from app.audio.normalize import remove_dc
from app.audio.resample import resample
from app.audio.validate import require_voice, validate_input
from app.config import Config
from app.models import SING, AlignmentMap, AudioFeatures, Flag, ProcessReport, ReferenceMap
from app.processing.backends import get_backend
from app.processing.pitch import PitchPlan, plan_correction
from app.quality.artifacts import protect_clipping
from app.quality.metrics import pitch_error
from app.quality.voice_similarity import voice_similarity


class AlignmentFailed(RuntimeError):
    def __init__(self, message: str, report: ProcessReport):
        super().__init__(message)
        self.report = report


@dataclass
class ProcessResult:
    audio: np.ndarray
    sample_rate: int
    report: ProcessReport
    user: AudioFeatures
    reference: ReferenceMap
    alignment: AlignmentMap
    plan: PitchPlan
    processed_features: AudioFeatures | None = None
    timings: dict[str, float] = field(default_factory=dict)


@contextmanager
def _timer(timings: dict[str, float], name: str):
    t0 = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t0, 2)


def _for_analysis(y: np.ndarray, cfg: Config) -> np.ndarray:
    return resample(y, cfg.audio.sample_rate, cfg.audio.analysis_sample_rate)


def process_audio(
    ref_y: np.ndarray,
    user_y: np.ndarray,
    cfg: Config,
    reference: ReferenceMap | None = None,
    evaluate: bool = True,
    speaker_model: bool = False,
) -> ProcessResult:
    """Run the pipeline on mono float32 arrays at cfg.audio.sample_rate."""
    sr = cfg.audio.sample_rate
    timings: dict[str, float] = {}
    report = ProcessReport(backend=cfg.backend, settings=cfg.to_dict())

    with _timer(timings, "analyze_reference"):
        if reference is None:
            reference, _ = analyze_reference(_for_analysis(ref_y, cfg), cfg)
    ref = reference.features
    with _timer(timings, "analyze_user"):
        user = analyze_signal(_for_analysis(user_y, cfg), cfg)
    require_voice(int((ref.labels == SING).sum()), ref.hop, "reference")
    require_voice(int((user.labels == SING).sum()), user.hop, "user")

    with _timer(timings, "align"):
        alignment = align(user, ref, cfg.alignment, cfg.vad.energy_db)
    report.alignment = {
        "cost_ratio": round(alignment.distance, 3),
        "offset_s": round(alignment.offset, 3),
        "user_active_s": [round(i * user.hop, 2) for i in alignment.user_range],
        "user_phrases": len(user.phrases),
        "reference_phrases": len(reference.phrases),
    }
    if alignment.failed:
        report.flags.append(
            Flag("ALIGNMENT_FAILED", detail=f"cost ratio {alignment.distance:.2f} > "
                                            f"{cfg.alignment.max_cost_ratio}")
        )
        report.timings = timings
        raise AlignmentFailed(
            "The user recording does not match the reference well enough to align "
            f"(cost ratio {alignment.distance:.2f} > {cfg.alignment.max_cost_ratio}).",
            report,
        )

    with _timer(timings, "pitch_plan"):
        ref_on_user = reference_cents_on_user(alignment, user, ref)
        user_cents = hz_to_cents(user.f0)
        user_cents[user.labels != SING] = np.nan
        octave, residual = detect_octave(user_cents, ref_on_user, cfg.pitch.octave_detect)
        plan = plan_correction(user, ref_on_user, octave, residual, cfg.pitch)
    report.pitch = plan.summary()
    report.flags.extend(plan.flags)

    with _timer(timings, f"resynthesis_{cfg.backend}"):
        backend = get_backend(cfg)
        out = backend.apply(user_y, sr, plan.correction_cents, user.hop, user)
    out, clip_flags = protect_clipping(out)
    report.flags.extend(clip_flags)

    processed = None
    if evaluate:
        with _timer(timings, "evaluate"):
            processed = analyze_signal(_for_analysis(out, cfg), cfg)
            proc_cents = hz_to_cents(processed.f0)
            proc_cents[processed.labels != SING] = np.nan
            n = min(len(proc_cents), len(plan.target_cents))
            active = (user.labels == SING)
            report.metrics = {
                "pitch_before": pitch_error(plan.user_cents, plan.target_cents),
                "pitch_after": pitch_error(proc_cents[:n], plan.target_cents[:n]),
                "voice": voice_similarity(user_y, out, sr, active, speaker_model),
            }

    report.timings = timings
    return ProcessResult(out, sr, report, user, reference, alignment, plan, processed, timings)


def process_files(
    reference_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    cfg: Config,
    cache_dir: Path | None = Path(".cache/analysis"),
    debug_dir: Path | None = None,
    evaluate: bool = True,
    speaker_model: bool = False,
) -> ProcessResult:
    sr = cfg.audio.sample_rate
    t0 = time.perf_counter()
    ref_y, ref_sr = load_audio(reference_path, sr)
    user_y, user_sr = load_audio(input_path, sr)
    warnings = validate_input(ref_y, sr, ref_sr, "reference")
    warnings += validate_input(user_y, sr, user_sr, "user")
    ref_y, user_y = remove_dc(ref_y), remove_dc(user_y)

    reference, cache_hit = analyze_reference(
        _for_analysis(ref_y, cfg), cfg, source=reference_path, cache_dir=cache_dir
    )
    try:
        result = process_audio(ref_y, user_y, cfg, reference, evaluate, speaker_model)
    except AlignmentFailed as exc:
        exc.report.reference, exc.report.input = str(reference_path), str(input_path)
        exc.report.flags = warnings + exc.report.flags
        _write_report(exc.report, Path(output_path))
        raise

    report = result.report
    report.reference, report.input = str(reference_path), str(input_path)
    report.flags = warnings + report.flags
    report.timings["reference_cache_hit"] = float(cache_hit)

    written = export_audio(result.audio, sr, output_path)
    report.output = str(written)
    report.timings["total"] = round(time.perf_counter() - t0, 2)
    _write_report(report, written)

    if debug_dir is not None:
        write_debug(result, ref_y, user_y, Path(debug_dir))
    return result


def report_path_for(output: Path) -> Path:
    return output.with_name(output.stem + ".report.json")


def _write_report(report: ProcessReport, output: Path) -> Path:
    path = report_path_for(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_debug(result: ProcessResult, ref_y: np.ndarray, user_y: np.ndarray, out: Path) -> None:
    from app.analysis.visualize import plot_alignment, plot_analysis, plot_pitch_correction

    out.mkdir(parents=True, exist_ok=True)
    sr = result.sample_rate
    user, ref = result.user, result.reference.features
    plot_analysis(user_y, sr, user, "User", out / "user_analysis.png")
    plot_analysis(ref_y, sr, ref, "Reference", out / "reference_analysis.png")
    plot_alignment(result.alignment, user.hop, out / "alignment.png")

    plan = result.plan
    curves = {
        "user": plan.user_cents / 100 + 69,
        "reference": plan.target_cents / 100 + 69,
        "target": (plan.user_cents + plan.correction_cents) / 100 + 69,
    }
    if result.processed_features is not None:
        proc = hz_to_cents(result.processed_features.f0)
        proc[result.processed_features.labels != SING] = np.nan
        curves["processed"] = proc / 100 + 69
    plot_pitch_correction(user.timestamps, curves, out / "pitch_correction.png")

    export_audio(result.audio, sr, out / "phase1_pitch.wav")
    (out / "reference_map.json").write_text(
        json.dumps(result.reference.to_json(), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out / "curves.npz",
        timestamps=user.timestamps,
        user_cents=plan.user_cents,
        target_cents=plan.target_cents,
        correction_cents=plan.correction_cents,
        ref_position=result.alignment.ref_position,
    )
