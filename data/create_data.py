"""Generate the synthetic ECG record used by the whole project.

The record is a *simulated* single-lead ECG: a repeating P-QRS-T complex built
from a sum of Gaussians, corrupted with the three artefacts that dominate real
ambulatory recordings (50 Hz powerline, low-frequency baseline wander and
broadband white noise).

Everything is deterministic: the same ``SEED`` always produces the same files,
so the numbers reported in ``README.md`` can be reproduced on any machine.

Outputs (paths relative to the repository root)
-----------------------------------------------
``data/raw/ecg_noisy_raw.csv``
    ``time, voltage`` - the contaminated signal (the "measurement").
``data/process/ecg_reference_clean.csv``
    ``time, voltage`` - the ground-truth signal *before* noise was added. This
    is the reference used to score every filter, and it is never shown to the
    filtering code at run time.
``data/process/ecg_metadata.json``
    Sampling rate, heart rate and noise parameters, so the app can quote the
    true values next to the estimated ones.

The fourth file, ``data/process/clean_ecg_noisy.csv``, is produced by
``src/fft_denoising.py`` (it is the *output* of the filter, not of the
generator) and is kept for backwards compatibility with older notebooks.

Usage
-----
    .venv/bin/python data/create_data.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESS_DIR = REPO_ROOT / "data" / "process"

# --- Recording parameters -------------------------------------------------
FS: float = 1000.0
"""Sampling rate in Hz (1 ms resolution, the usual rate for diagnostic ECG)."""

DURATION_S: float = 10.0
"""Record length in seconds."""

TARGET_BPM: float = 72.0
"""Nominal heart rate. 72 bpm sits inside the requested 60-75 bpm band."""

RR_JITTER: float = 0.02
"""Beat-to-beat RR variability as a fraction of the mean RR interval."""

SEED: int = 20240501
"""Seed for ``np.random.default_rng``; changing it changes the record."""

# --- Artefact parameters --------------------------------------------------
POWERLINE_HZ: float = 50.0
POWERLINE_AMP: float = 0.35
"""Peak amplitude of the mains hum, in the same units as the R peak (1.0)."""

WANDER_HZ: float = 0.3
WANDER_AMP: float = 0.15
"""Baseline wander: slow respiration-scale drift."""

WHITE_NOISE_STD: float = 0.03
"""Standard deviation of the additive broadband noise."""


@dataclass(frozen=True)
class WaveComponent:
    """One Gaussian bump of the P-QRS-T complex.

    Attributes:
        name: Label used in plots and documentation.
        offset_s: Centre of the bump relative to the R peak, in seconds.
        amplitude: Peak height in normalised voltage units.
        sigma_s: Standard deviation of the Gaussian, in seconds.
    """

    name: str
    offset_s: float
    amplitude: float
    sigma_s: float


PQRST: tuple[WaveComponent, ...] = (
    WaveComponent("P", -0.20, 0.15, 0.025),
    WaveComponent("Q", -0.035, -0.10, 0.008),
    WaveComponent("R", 0.0, 1.0, 0.010),
    WaveComponent("S", 0.035, -0.20, 0.010),
    WaveComponent("T", 0.26, 0.30, 0.045),
)
"""Morphology of one beat, expressed as Gaussians around the R peak."""


@dataclass(frozen=True)
class SyntheticECG:
    """A generated record plus the parameters needed to score a filter.

    Attributes:
        time: Sample times in seconds, shape ``(n,)``.
        clean: Ground-truth voltage, shape ``(n,)``.
        noisy: Contaminated voltage, shape ``(n,)``.
        beat_times: True R-peak times in seconds, shape ``(n_beats,)``.
        fs: Sampling rate in Hz.
        bpm: Mean heart rate actually realised in the record.
    """

    time: np.ndarray
    clean: np.ndarray
    noisy: np.ndarray
    beat_times: np.ndarray
    fs: float
    bpm: float

    @property
    def duration_s(self) -> float:
        """Length of the record in seconds."""
        return float(self.time[-1] - self.time[0])


def _gaussian(t: np.ndarray, centre: float, amplitude: float, sigma: float) -> np.ndarray:
    """Return a Gaussian bump sampled on ``t``.

    Args:
        t: Sample times in seconds.
        centre: Centre of the bump in seconds.
        amplitude: Peak height.
        sigma: Standard deviation in seconds.

    Returns:
        Array with the same shape as ``t``.
    """
    return amplitude * np.exp(-0.5 * ((t - centre) / sigma) ** 2)


def beat_times(duration_s: float, bpm: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Place R peaks on a (slightly irregular) rhythm.

    The first beat sits half a cycle into the record so that the P wave of the
    first complex is not clipped by the left edge.

    Args:
        duration_s: Record length in seconds.
        bpm: Mean heart rate.
        jitter: RR variability as a fraction of the mean RR interval.
        rng: Seeded random generator.

    Returns:
        Sorted R-peak times in seconds.
    """
    mean_rr = 60.0 / bpm
    times: list[float] = []
    t_current = mean_rr * 0.5
    while t_current < duration_s - 0.1:
        times.append(t_current)
        step = mean_rr * (1.0 + jitter * float(rng.standard_normal()))
        t_current += float(np.clip(step, 0.5 * mean_rr, 1.5 * mean_rr))
    return np.asarray(times, dtype=float)


def clean_ecg(t: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Build a clean ECG by summing one P-QRS-T complex per beat.

    Args:
        t: Sample times in seconds.
        peaks: R-peak times in seconds.

    Returns:
        Clean voltage trace with the same shape as ``t``.
    """
    signal = np.zeros_like(t, dtype=float)
    for peak in peaks:
        for wave in PQRST:
            signal += _gaussian(t, float(peak) + wave.offset_s, wave.amplitude, wave.sigma_s)
    return signal


def add_artefacts(
    clean: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add powerline hum, baseline wander and white noise to a clean trace.

    Args:
        clean: Clean voltage trace.
        t: Sample times in seconds.
        rng: Seeded random generator.

    Returns:
        Contaminated voltage trace.
    """
    powerline = POWERLINE_AMP * np.sin(2.0 * np.pi * POWERLINE_HZ * t + rng.uniform(0.0, 2.0 * np.pi))
    wander = WANDER_AMP * np.sin(2.0 * np.pi * WANDER_HZ * t + rng.uniform(0.0, 2.0 * np.pi))
    white = rng.normal(0.0, WHITE_NOISE_STD, size=t.shape)
    return clean + powerline + wander + white


def generate(
    fs: float = FS,
    duration_s: float = DURATION_S,
    bpm: float = TARGET_BPM,
    seed: int = SEED,
) -> SyntheticECG:
    """Generate one synthetic ECG record.

    Args:
        fs: Sampling rate in Hz.
        duration_s: Record length in seconds.
        bpm: Mean heart rate.
        seed: Seed for the random generator.

    Returns:
        A :class:`SyntheticECG` holding the clean and contaminated traces.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    t = np.arange(n_samples, dtype=float) / fs
    peaks = beat_times(duration_s, bpm, RR_JITTER, rng)
    clean = clean_ecg(t, peaks)
    noisy = add_artefacts(clean, t, rng)
    realised_bpm = 60.0 / float(np.mean(np.diff(peaks))) if peaks.size > 1 else bpm
    return SyntheticECG(
        time=t,
        clean=clean,
        noisy=noisy,
        beat_times=peaks,
        fs=fs,
        bpm=realised_bpm,
    )


def save(record: SyntheticECG, raw_dir: Path = RAW_DIR, process_dir: Path = PROCESS_DIR) -> dict[str, Path]:
    """Write the record and its metadata to disk.

    Args:
        record: The generated record.
        raw_dir: Directory for the contaminated signal.
        process_dir: Directory for the reference signal and metadata.

    Returns:
        Mapping of a short label to the path that was written.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    process_dir.mkdir(parents=True, exist_ok=True)

    noisy_path = raw_dir / "ecg_noisy_raw.csv"
    reference_path = process_dir / "ecg_reference_clean.csv"
    metadata_path = process_dir / "ecg_metadata.json"

    pd.DataFrame({"time": record.time, "voltage": record.noisy}).to_csv(
        noisy_path, index=False, float_format="%.6f"
    )
    pd.DataFrame({"time": record.time, "voltage": record.clean}).to_csv(
        reference_path, index=False, float_format="%.6f"
    )

    metadata = {
        "source": "synthetic",
        "generator": "data/create_data.py",
        "seed": SEED,
        "fs_hz": record.fs,
        "duration_s": record.duration_s,
        "n_samples": int(record.time.size),
        "target_bpm": TARGET_BPM,
        "realised_bpm": round(record.bpm, 4),
        "n_beats": int(record.beat_times.size),
        "beat_times_s": [round(float(b), 4) for b in record.beat_times],
        "artefacts": {
            "powerline": {"frequency_hz": POWERLINE_HZ, "amplitude": POWERLINE_AMP},
            "baseline_wander": {"frequency_hz": WANDER_HZ, "amplitude": WANDER_AMP},
            "white_noise": {"std": WHITE_NOISE_STD},
        },
        "morphology": [asdict(wave) for wave in PQRST],
        "units": "normalised voltage (R peak = 1.0), time in seconds",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"noisy": noisy_path, "reference": reference_path, "metadata": metadata_path}


def main() -> None:
    """Generate the record, write the files and print a short summary."""
    record = generate()
    paths = save(record)
    print("Synthetic ECG written:")
    for label, path in paths.items():
        print(f"  {label:<10} {path.relative_to(REPO_ROOT)}")
    print(
        f"  {record.time.size} samples @ {record.fs:.0f} Hz, "
        f"{record.bpm:.2f} bpm over {record.duration_s:.1f} s"
    )


if __name__ == "__main__":
    main()
