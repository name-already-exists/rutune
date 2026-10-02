"""Command line interface: `python -m app.cli <command>` or `rutune <command>`."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

import typer

from app.config import PRESETS, load_config

cli = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@cli.command()
def process(
    reference: Path = typer.Option(..., "--reference", "-r", help="Reference (sample) vocal"),
    input: Path = typer.Option(..., "--input", "-i", help="User vocal"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file (.wav/.flac/.mp3)"),
    strength: str | None = typer.Option(
        None, "--strength", "-s", help=f"Preset: {' | '.join(PRESETS)}"
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config file"),
    backend: str | None = typer.Option(None, "--backend", help="world | rubberband"),
    debug_dir: Path | None = typer.Option(None, "--debug-dir", help="Write plots/curves here"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not cache reference analysis"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip post-processing metrics"),
    speaker_model: bool = typer.Option(
        False, "--speaker-model", help="Also compute ECAPA speaker similarity (needs [speaker])"
    ),
):
    """Process a user vocal so its performance follows the reference."""
    from app.audio.validate import InputError
    from app.pipeline.processor import AlignmentFailed, process_files, report_path_for

    cfg = load_config(config, strength)
    if backend:
        cfg.backend = backend
    try:
        result = process_files(
            reference, input, output, cfg,
            cache_dir=None if no_cache else Path(".cache/analysis"),
            debug_dir=debug_dir,
            evaluate=not no_eval,
            speaker_model=speaker_model,
        )
    except (InputError, AlignmentFailed) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    report = result.report
    typer.secho(f"Wrote {report.output}", fg=typer.colors.GREEN)
    typer.echo(f"Report: {report_path_for(Path(report.output))}")
    m = report.metrics
    if m:
        b, a = m["pitch_before"], m["pitch_after"]
        typer.echo(
            f"Pitch MAE: {b['mae_cents']} -> {a['mae_cents']} cents | "
            f"within 50c: {b['within_50c']} -> {a['within_50c']}"
        )
        typer.echo(f"Voice: {json.dumps(m['voice'])}")
    typer.echo(f"Alignment: {json.dumps(report.alignment)}")
    for flag in report.flags:
        where = f" [{flag.start:.2f}-{flag.end:.2f}s]" if flag.start is not None else ""
        typer.secho(f"  ! {flag.code}{where} {flag.detail}", fg=typer.colors.YELLOW)
    typer.echo(f"Time: {report.timings.get('total')}s")


@cli.command()
def analyze(
    files: list[Path] = typer.Argument(..., help="Audio files to analyse"),
    out_dir: Path = typer.Option(Path("samples/output"), "--out-dir", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
):
    """Phase 0 playground: waveform/spectrogram/F0/phrase plots + JSON map per file."""
    from app.analysis.analyzer import analyze_signal
    from app.analysis.reference import to_reference_map
    from app.analysis.visualize import plot_analysis
    from app.audio.loader import load_audio
    from app.audio.resample import resample

    cfg = load_config(config)
    for path in files:
        y, _ = load_audio(path, cfg.audio.sample_rate)
        feats = analyze_signal(resample(y, cfg.audio.sample_rate, cfg.audio.analysis_sample_rate),
                               cfg)
        png = plot_analysis(y, cfg.audio.sample_rate, feats, path.stem,
                            out_dir / f"{path.stem}_analysis.png")
        js = out_dir / f"{path.stem}_analysis.json"
        js.write_text(json.dumps(to_reference_map(feats).to_json(), indent=2), encoding="utf-8")
        n_notes = sum(len(p.notes) for p in feats.phrases)
        typer.echo(f"{path.name}: {feats.duration:.1f}s, {len(feats.phrases)} phrases, "
                   f"{n_notes} notes -> {png}, {js}")


@cli.command()
def doctor():
    """Check the runtime environment."""
    ok = True
    typer.echo(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info >= (3, 14):
        typer.secho("  ! Python 3.14 is not supported by numba/torch yet; use 3.11-3.13",
                    fg=typer.colors.YELLOW)
    required = ["numpy", "scipy", "librosa", "soundfile", "pyworld", "pyloudnorm", "yaml",
                "matplotlib"]
    optional = ["torchcrepe", "pyrubberband", "speechbrain"]
    for name in required + optional:
        try:
            mod = importlib.import_module(name)
            typer.echo(f"  ok  {name} {getattr(mod, '__version__', '')}")
        except ImportError:
            is_required = name in required
            ok &= not is_required
            typer.secho(f"  --  {name} ({'REQUIRED' if is_required else 'optional'})",
                        fg=typer.colors.RED if is_required else typer.colors.YELLOW)
    for tool, why in (("ffmpeg", "mp3/aac decode + export"),
                      ("rubberband", "backend 'rubberband' only")):
        path = shutil.which(tool)
        typer.echo(f"  {'ok' if path else '--'}  {tool}: {path or 'not found'} ({why})")
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
