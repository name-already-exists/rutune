"""Diagnostic plots (waveform, spectrogram, F0, VAD, alignment)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from app.models import BREATH, SING, UNVOICED, AlignmentMap, AudioFeatures

_LABEL_COLORS = {SING: "tab:green", UNVOICED: "tab:orange", BREATH: "tab:purple"}


def plot_analysis(y: np.ndarray, sr: int, feats: AudioFeatures, title: str, path: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    t = np.arange(len(y)) / sr

    axes[0].plot(t, y, lw=0.4, color="0.3")
    for label, color in _LABEL_COLORS.items():
        mask = feats.labels == label
        axes[0].fill_between(
            feats.timestamps, -1, 1, where=mask, color=color, alpha=0.15, step="mid", lw=0
        )
    for p in feats.phrases:
        axes[0].axvspan(p.start, p.end, ymin=0.95, ymax=1.0, color="tab:blue", alpha=0.6)
    axes[0].set_ylim(-1, 1)
    axes[0].set_title(f"{title} — waveform (green=sing, orange=unvoiced, purple=breath)")

    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(
        spec, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=axes[1], cmap="magma"
    )
    axes[1].plot(feats.timestamps, feats.f0, color="cyan", lw=1.0)
    axes[1].set_ylim(50, 4000)
    axes[1].set_title("Spectrogram + F0")

    axes[2].plot(feats.timestamps, feats.midi, lw=1.0, label="F0 (MIDI)")
    for p in feats.phrases:
        for n in p.notes:
            m = 69 + 12 * np.log2(n.f0 / 440.0)
            axes[2].hlines(m, n.start, n.end, color="tab:red", lw=2)
    axes[2].set_ylabel("MIDI note")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"Notes ({sum(len(p.notes) for p in feats.phrases)}) / phrases "
                      f"({len(feats.phrases)})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_pitch_correction(
    timestamps: np.ndarray,
    curves: dict[str, np.ndarray],
    path: Path,
) -> Path:
    """Overlay of user / mapped reference / target / processed F0 (cents or MIDI)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    styles = {
        "user": {"color": "0.5", "lw": 1.0},
        "reference": {"color": "tab:blue", "lw": 1.0, "alpha": 0.8},
        "target": {"color": "tab:red", "lw": 1.0, "ls": "--"},
        "processed": {"color": "tab:green", "lw": 1.0},
    }
    for name, values in curves.items():
        ax.plot(timestamps[: len(values)], values, label=name, **styles.get(name, {}))
    ax.set_xlabel("User time (s)")
    ax.set_ylabel("MIDI note")
    ax.legend(loc="upper right")
    ax.set_title("Pitch correction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_alignment(alignment: AlignmentMap, hop: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(alignment.reference_indices * hop, alignment.user_indices * hop, ",", color="0.6")
    ax.plot(alignment.ref_position * hop, np.arange(len(alignment.ref_position)) * hop,
            color="tab:red", lw=1.0)
    ax.set_xlabel("Reference time (s)")
    ax.set_ylabel("User time (s)")
    ax.set_title(f"Alignment (cost ratio {alignment.distance:.2f})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
