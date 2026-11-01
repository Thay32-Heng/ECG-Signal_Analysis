"""ECG denoising pipeline: FFT notch, zero-phase bandpass, scoring and QRS detection.

This module *is* the project's "model". There is no machine learning here: the
model is a deterministic signal-processing chain, and the numbers that matter
are how much signal-to-noise ratio it recovers and whether the heartbeat
survives the filtering.

Chain
-----
1. :func:`notch` removes mains hum (50 or 60 Hz) by attenuating the matching
   FFT bins with a raised-cosine taper, which avoids the Gibbs ringing produced
   by the hard "set the bin to zero" version this project started with.
2. :func:`bandpass` is a Butterworth band-pass applied forwards and backwards
   (``sosfiltfilt``), so the result is zero-phase: the R peaks do not move.
3. :func:`snr` scores any candidate against the known clean reference.
4. :func:`qrs_indices` / :func:`estimate_bpm` recover the heart rate with a
   Pan-Tompkins-lite detector (5-15 Hz, derivative, square, moving-window
   integration, adaptive threshold, 200 ms refractory).

Running this file as a script writes ``data/process/metrics.csv`` and the
backwards-compatible ``data/process/clean_ecg_noisy.csv``.

Usage
-----
    .venv/bin/python src/fft_denoising.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "ecg_noisy_raw.csv"
REFERENCE_PATH = REPO_ROOT / "data" / "process" / "ecg_reference_clean.csv"
METADATA_PATH = REPO_ROOT / "data" / "process" / "ecg_metadata.json"
METRICS_PATH = REPO_ROOT / "data" / "process" / "metrics.csv"
CLEANED_PATH = REPO_ROOT / "data" / "process" / "clean_ecg_noisy.csv"

DEFAULT_FS: float = 1000.0
DEFAULT_LOW_HZ: float = 0.5
DEFAULT_HIGH_HZ: float = 40.0
DEFAULT_Q: float = 35.0
DEFAULT_ORDER: int = 4

QRS_BAND_HZ: tuple[float, float] = (5.0, 15.0)
REFRACTORY_S: float = 0.20
INTEGRATION_S: float = 0.15


# --------------------------------------------------------------------------
# Spectrum
# --------------------------------------------------------------------------
def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single-sided amplitude spectrum.

    The magnitude is scaled by ``2 / n`` so that a pure sine of amplitude ``A``
    shows a peak at ``A`` - this makes the plots readable in the signal's own
    units instead of arbitrary FFT counts.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.

    Returns:
        Tuple ``(freq, magnitude)`` of length ``n // 2 + 1``.
    """
    samples = np.asarray(x, dtype=float)
    if samples.size == 0:
        raise ValueError("spectrum() received an empty signal")
    magnitude = np.abs(np.fft.rfft(samples)) * 2.0 / samples.size
    freq = np.fft.rfftfreq(samples.size, d=1.0 / fs)
    return freq, magnitude


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def notch(x: np.ndarray, fs: float, f0: float, q: float = DEFAULT_Q) -> np.ndarray:
    """Attenuate a narrow band around ``f0`` in the frequency domain.

    Bins inside ``half_width = max(1 Hz, f0 / (2q))`` are multiplied by a
    raised-cosine gain that reaches 0 at ``f0`` and returns smoothly to 1 at the
    band edge. The smooth taper is deliberate: zeroing a bin abruptly is what
    turned the original committed "cleaned" file into Gibbs ringing.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.
        f0: Centre frequency to remove (50 or 60 Hz for mains hum).
        q: Quality factor controlling the notch width.

    Returns:
        Filtered signal with the same shape as ``x``.
    """
    samples = np.asarray(x, dtype=float)
    n = samples.size
    if n == 0:
        raise ValueError("notch() received an empty signal")
    if not 0.0 < f0 < fs / 2.0:
        raise ValueError(f"notch frequency {f0} Hz is outside (0, {fs / 2.0}) Hz")

    half_width = max(1.0, f0 / (2.0 * q))
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    distance = np.abs(freq - f0)
    gain = np.ones_like(freq)
    inside = distance <= half_width
    gain[inside] = 0.5 * (1.0 - np.cos(np.pi * distance[inside] / half_width))

    return np.fft.irfft(np.fft.rfft(samples) * gain, n=n)


def bandpass(
    x: np.ndarray,
    fs: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    ``sosfiltfilt`` runs the filter forwards and backwards, so the phase
    response is exactly zero and the R peaks keep their original timing.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.
        low: Lower cutoff in Hz (removes baseline wander).
        high: Upper cutoff in Hz (removes high-frequency noise).
        order: Butterworth order; 4 gives a 24 dB/octave roll-off per side.

    Returns:
        Filtered signal with the same shape as ``x``.
    """
    samples = np.asarray(x, dtype=float)
    nyquist = fs / 2.0
    if not 0.0 < low < high < nyquist:
        raise ValueError(f"invalid band {low}-{high} Hz for fs={fs} Hz (Nyquist {nyquist} Hz)")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, samples)


def denoise(
    x: np.ndarray,
    fs: float,
    notch_hz: float | None = 50.0,
    use_bandpass: bool = True,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
    q: float = DEFAULT_Q,
) -> np.ndarray:
    """Run the full denoising chain: optional notch, then optional band-pass.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.
        notch_hz: Mains frequency to notch out, or ``None`` to skip the notch.
        use_bandpass: Whether to apply the Butterworth band-pass.
        low: Band-pass lower cutoff in Hz.
        high: Band-pass upper cutoff in Hz.
        order: Band-pass filter order.
        q: Notch quality factor.

    Returns:
        Filtered signal with the same shape as ``x``.
    """
    filtered = np.asarray(x, dtype=float)
    if notch_hz is not None:
        filtered = notch(filtered, fs, notch_hz, q=q)
    if use_bandpass:
        filtered = bandpass(filtered, fs, low=low, high=high, order=order)
    return filtered


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio of ``estimate`` against the clean reference, in dB.

    ``SNR = 10 log10(var(reference) / var(reference - estimate))``. The residual
    is treated as noise, so a perfect reconstruction returns ``+inf``.

    Args:
        reference: Ground-truth signal.
        estimate: Candidate reconstruction of the same length.

    Returns:
        SNR in decibels.
    """
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs estimate {est.shape}")
    residual_var = float(np.var(ref - est))
    signal_var = float(np.var(ref))
    if residual_var <= 0.0:
        return float("inf")
    if signal_var <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal_var / residual_var))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Root-mean-square error between two signals of the same length."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs estimate {est.shape}")
    return float(np.sqrt(np.mean((ref - est) ** 2)))


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson correlation between two signals; 1.0 means identical shape."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs estimate {est.shape}")
    if ref.std() == 0.0 or est.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ref, est)[0, 1])


# --------------------------------------------------------------------------
# QRS detection (Pan-Tompkins-lite)
# --------------------------------------------------------------------------
def _integrated_envelope(x: np.ndarray, fs: float) -> np.ndarray:
    """Build the Pan-Tompkins detection envelope.

    Band-pass 5-15 Hz (the band where QRS energy dominates), differentiate,
    square, then integrate over a 150 ms moving window.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.

    Returns:
        Non-negative envelope with the same length as ``x``.
    """
    samples = np.asarray(x, dtype=float)
    sos = signal.butter(2, list(QRS_BAND_HZ), btype="bandpass", fs=fs, output="sos")
    qrs_band = signal.sosfiltfilt(sos, samples)
    derivative = np.gradient(qrs_band)
    squared = derivative**2
    window = max(1, int(round(INTEGRATION_S * fs)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(squared, kernel, mode="same")


def qrs_indices(
    x: np.ndarray,
    fs: float,
    refractory_s: float = REFRACTORY_S,
) -> np.ndarray:
    """Locate R peaks with an adaptive-threshold Pan-Tompkins-lite detector.

    Candidate peaks are taken from the integrated envelope, then accepted or
    rejected one by one against a threshold that tracks the running signal and
    noise peak levels (the SPKI/NPKI scheme from Pan & Tompkins, 1985). Each
    accepted beat is snapped to the local maximum of the 5-15 Hz band around
    it, which is where the QRS energy actually sits.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.
        refractory_s: Minimum distance between beats in seconds.

    Returns:
        Sorted sample indices of the detected R peaks.
    """
    samples = np.asarray(x, dtype=float)
    if samples.size < int(fs * 0.5):
        return np.empty(0, dtype=int)

    envelope = _integrated_envelope(samples, fs)
    min_distance = max(1, int(round(refractory_s * fs)))
    candidates, _ = signal.find_peaks(envelope, distance=min_distance)
    if candidates.size == 0:
        return np.empty(0, dtype=int)

    signal_level = 0.25 * float(envelope[candidates].max())
    noise_level = 0.05 * signal_level
    threshold = noise_level + 0.25 * (signal_level - noise_level)

    accepted: list[int] = []
    for index in candidates:
        value = float(envelope[index])
        if value > threshold:
            accepted.append(int(index))
            signal_level = 0.125 * value + 0.875 * signal_level
        else:
            noise_level = 0.125 * value + 0.875 * noise_level
        threshold = noise_level + 0.25 * (signal_level - noise_level)

    if not accepted:
        return np.empty(0, dtype=int)

    # Snap each detection onto the local QRS maximum.
    sos = signal.butter(2, list(QRS_BAND_HZ), btype="bandpass", fs=fs, output="sos")
    qrs_band = signal.sosfiltfilt(sos, samples)
    search = max(1, int(round(0.10 * fs)))
    peaks: list[int] = []
    for index in accepted:
        low = max(0, index - search)
        high = min(qrs_band.size, index + search + 1)
        peaks.append(low + int(np.argmax(np.abs(qrs_band[low:high]))))

    return np.unique(np.asarray(peaks, dtype=int))


def rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak sample indices to RR intervals in seconds."""
    index = np.asarray(peaks, dtype=int)
    if index.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(index) / fs


def estimate_bpm(x: np.ndarray, fs: float, refractory_s: float = REFRACTORY_S) -> float:
    """Estimate heart rate in beats per minute from a single-lead ECG.

    The rate is the median RR interval converted to bpm. Intervals outside
    0.3-2.0 s (i.e. outside 30-200 bpm) are dropped as missed or spurious
    detections, which keeps one bad beat from destroying the estimate.

    Args:
        x: Signal samples.
        fs: Sampling rate in Hz.
        refractory_s: Minimum distance between beats in seconds.

    Returns:
        Heart rate in bpm, or ``nan`` when fewer than two beats are detected.
    """
    peaks = qrs_indices(x, fs, refractory_s=refractory_s)
    intervals = rr_intervals(peaks, fs)
    if intervals.size == 0:
        return float("nan")
    plausible = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    if plausible.size == 0:
        return float("nan")
    return float(60.0 / float(np.median(plausible)))


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One filter configuration to score.

    Attributes:
        name: Label used in the metrics table and in the app.
        notch_hz: Mains frequency to notch, or ``None``.
        use_bandpass: Whether the Butterworth band-pass is applied.
    """

    name: str
    notch_hz: float | None
    use_bandpass: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("raw", None, False),
    Variant("notch_only", 50.0, False),
    Variant("bandpass_only", None, True),
    Variant("notch+bandpass", 50.0, True),
)
"""Configurations compared in ``metrics.csv``."""


def read_metadata(path: Path = METADATA_PATH) -> dict[str, object]:
    """Load the generator metadata, falling back to defaults when absent.

    Args:
        path: Location of ``ecg_metadata.json``.

    Returns:
        Parsed metadata, or a minimal default mapping.
    """
    if not path.exists():
        return {"fs_hz": DEFAULT_FS, "source": "synthetic"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_pair(
    raw_path: Path = RAW_PATH,
    reference_path: Path = REFERENCE_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the noisy record and its clean reference.

    Args:
        raw_path: CSV with ``time, voltage`` for the contaminated signal.
        reference_path: CSV with ``time, voltage`` for the ground truth.

    Returns:
        Tuple ``(time, noisy, reference, fs)``; ``fs`` is inferred from the
        time column.
    """
    noisy_df = pd.read_csv(raw_path)
    reference_df = pd.read_csv(reference_path)
    if len(noisy_df) != len(reference_df):
        raise ValueError(
            f"noisy and reference records differ in length ({len(noisy_df)} vs {len(reference_df)})"
        )
    time = noisy_df["time"].to_numpy(dtype=float)
    step = float(np.median(np.diff(time))) if time.size > 1 else 1.0 / DEFAULT_FS
    fs = 1.0 / step
    return time, noisy_df["voltage"].to_numpy(dtype=float), reference_df["voltage"].to_numpy(dtype=float), fs


def evaluate(
    reference: np.ndarray,
    noisy: np.ndarray,
    fs: float,
    variant: Variant,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    q: float = DEFAULT_Q,
) -> dict[str, float | str]:
    """Score one filter variant against the clean reference.

    Args:
        reference: Ground-truth signal.
        noisy: Contaminated input signal.
        fs: Sampling rate in Hz.
        variant: Configuration to apply.
        low: Band-pass lower cutoff in Hz.
        high: Band-pass upper cutoff in Hz.
        q: Notch quality factor.

    Returns:
        One metrics row (see ``METRICS_COLUMNS``).
    """
    filtered = denoise(
        noisy,
        fs,
        notch_hz=variant.notch_hz,
        use_bandpass=variant.use_bandpass,
        low=low,
        high=high,
        q=q,
    )
    snr_in = snr(reference, noisy)
    snr_out = snr(reference, filtered)
    ref_peak = float(np.max(np.abs(reference)))
    out_peak = float(np.max(np.abs(filtered)))
    return {
        "variant": variant.name,
        "snr_in_db": round(snr_in, 2),
        "snr_out_db": round(snr_out, 2),
        "gain_db": round(snr_out - snr_in, 2),
        "bpm_before": round(estimate_bpm(noisy, fs), 2),
        "bpm_after": round(estimate_bpm(filtered, fs), 2),
        "correlation": round(correlation(reference, filtered), 4),
        "r_peak_ratio": round(out_peak / ref_peak, 4) if ref_peak else float("nan"),
        "std_ratio": round(float(np.std(filtered) / np.std(reference)), 4),
        "rmse": round(rmse(reference, filtered), 5),
    }


METRICS_COLUMNS: tuple[str, ...] = (
    "variant",
    "snr_in_db",
    "snr_out_db",
    "gain_db",
    "bpm_before",
    "bpm_after",
    "correlation",
    "r_peak_ratio",
    "std_ratio",
    "rmse",
)


def run_all() -> tuple[pd.DataFrame, np.ndarray, float]:
    """Score every variant and write the metrics table plus the cleaned record.

    Returns:
        Tuple ``(metrics, cleaned_signal, fs)`` where ``metrics`` has one row per
        variant and ``cleaned_signal`` is the production configuration
        (50 Hz notch + 0.5-40 Hz band-pass).
    """
    time, noisy, reference, fs = load_pair()
    rows = [evaluate(reference, noisy, fs, variant) for variant in VARIANTS]
    metrics = pd.DataFrame(rows, columns=list(METRICS_COLUMNS))

    production = denoise(noisy, fs, notch_hz=50.0, use_bandpass=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.DataFrame({"time": time, "voltage_cleaned": production}).to_csv(
        CLEANED_PATH, index=False, float_format="%.6f"
    )
    return metrics, production, fs


def main() -> None:
    """Run the comparison and print a compact report."""
    metrics, _, fs = run_all()
    print(f"ECG denoising metrics ({fs:.0f} Hz)\n")
    print(metrics.to_string(index=False))
    print(f"\nwritten: {METRICS_PATH.relative_to(REPO_ROOT)}")
    print(f"written: {CLEANED_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
