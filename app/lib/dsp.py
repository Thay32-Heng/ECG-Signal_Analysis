"""Thin, cached wrappers over :mod:`src.fft_denoising`.

Pages never call scipy directly: they call these functions, which add Streamlit
caching (so moving a slider back to a previous value is instant) and bundle the
metrics into typed objects for the metric cards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from src import fft_denoising as core

@dataclass(frozen=True)
class FilterSettings:
    """Everything the sidebar lets the user change.

    Attributes:
        fs: Sampling rate in Hz used by the filters and the detector.
        notch_hz: Mains frequency to remove, or ``None`` for no notch.
        use_bandpass: Whether the Butterworth band-pass runs.
        low: Band-pass lower cutoff in Hz.
        high: Band-pass upper cutoff in Hz.
        q: Notch quality factor.
    """

    fs: float
    notch_hz: float | None
    use_bandpass: bool
    low: float
    high: float
    q: float

    @property
    def label(self) -> str:
        """Short human-readable description of the chain."""
        parts = []
        parts.append(f"notch {self.notch_hz:.0f} Hz" if self.notch_hz else "no notch")
        parts.append(f"band-pass {self.low:g}-{self.high:g} Hz" if self.use_bandpass else "no band-pass")
        return " + ".join(parts)


@dataclass(frozen=True)
class SignalMetrics:
    """Scoring results for one record and one filter configuration.

    Attributes:
        snr_in_db: SNR of the raw record against the reference (``nan`` if unknown).
        snr_out_db: SNR of the filtered record against the reference.
        gain_db: ``snr_out_db - snr_in_db``.
        noise_reduction_db: ``10·log10(var(raw) / var(removed))`` — how far the removed component sits
            below the raw power. Usable without a reference.
        bpm_in: Heart rate estimated from the raw record.
        bpm_out: Heart rate estimated from the filtered record.
        true_bpm: Generator heart rate, when the record is the bundled one.
        n_beats_in: Beats detected in the raw record.
        n_beats_out: Beats detected in the filtered record.
        correlation: Pearson correlation with the reference (morphology).
        r_peak_ratio: Filtered R-peak height divided by the reference's.
        std_ratio: Filtered standard deviation divided by the reference's.
        rmse: Root-mean-square error against the reference.
    """

    snr_in_db: float
    snr_out_db: float
    gain_db: float
    noise_reduction_db: float
    bpm_in: float
    bpm_out: float
    true_bpm: float | None
    n_beats_in: int
    n_beats_out: int
    correlation: float
    r_peak_ratio: float
    std_ratio: float
    rmse: float

    @property
    def has_reference(self) -> bool:
        """Whether reference-based metrics are meaningful."""
        return bool(np.isfinite(self.snr_in_db) and np.isfinite(self.snr_out_db))

    @property
    def bpm_error(self) -> float | None:
        """Absolute BPM error against the generator, or ``None``."""
        if self.true_bpm is None or not np.isfinite(self.bpm_out):
            return None
        return abs(self.bpm_out - self.true_bpm)


@st.cache_data(show_spinner=False, max_entries=16)
def apply_filter(
    voltage: np.ndarray,
    fs: float,
    notch_hz: float | None,
    use_bandpass: bool,
    low: float,
    high: float,
    q: float,
) -> np.ndarray:
    """Run the denoising chain on an array.

    Args:
        voltage: Signal samples.
        fs: Sampling rate in Hz.
        notch_hz: Mains frequency to remove, or ``None``.
        use_bandpass: Whether to apply the band-pass.
        low: Band-pass lower cutoff in Hz.
        high: Band-pass upper cutoff in Hz.
        q: Notch quality factor.

    Returns:
        Filtered signal.
    """
    return core.denoise(
        voltage,
        fs,
        notch_hz=notch_hz,
        use_bandpass=use_bandpass,
        low=low,
        high=high,
        q=q,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def detect_qrs(voltage: np.ndarray, fs: float) -> np.ndarray:
    """Detect R peaks in an array.

    Args:
        voltage: Signal samples.
        fs: Sampling rate in Hz.

    Returns:
        Sample indices of the detected R peaks.
    """
    return core.qrs_indices(voltage, fs)


@st.cache_data(show_spinner=False, max_entries=16)
def spectrum_pair(
    voltage: np.ndarray,
    filtered: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the amplitude spectrum of the raw and filtered signals.

    Args:
        voltage: Raw samples.
        filtered: Filtered samples.
        fs: Sampling rate in Hz.

    Returns:
        Tuple ``(freq, magnitude_raw, magnitude_filtered)``.
    """
    freq, raw_mag = core.spectrum(voltage, fs)
    _, filtered_mag = core.spectrum(filtered, fs)
    return freq, raw_mag, filtered_mag


def measure(
    voltage: np.ndarray,
    filtered: np.ndarray,
    reference: np.ndarray | None,
    fs: float,
    true_bpm: float | None,
) -> SignalMetrics:
    """Score a filtered signal against the raw record and the reference.

    Args:
        voltage: Raw samples.
        filtered: Filtered samples.
        reference: Ground truth, or ``None`` for uploads without one.
        fs: Sampling rate in Hz.
        true_bpm: Generator heart rate when known.

    Returns:
        Populated :class:`SignalMetrics`.
    """
    peaks_in = detect_qrs(voltage, fs)
    peaks_out = detect_qrs(filtered, fs)

    if reference is not None and reference.shape == filtered.shape:
        snr_in = core.snr(reference, voltage)
        snr_out = core.snr(reference, filtered)
        correlation = core.correlation(reference, filtered)
        ref_peak = float(np.max(np.abs(reference)))
        r_peak_ratio = float(np.max(np.abs(filtered)) / ref_peak) if ref_peak else float("nan")
        std_ratio = float(np.std(filtered) / np.std(reference))
        error = core.rmse(reference, filtered)
    else:
        snr_in = float("nan")
        snr_out = float("nan")
        correlation = float("nan")
        r_peak_ratio = float("nan")
        std_ratio = float("nan")
        error = float("nan")

    removed = voltage - filtered
    noise_reduction = (
        float(10.0 * np.log10(np.var(voltage) / np.var(removed))) if np.var(removed) > 0 else float("nan")
    )

    return SignalMetrics(
        snr_in_db=snr_in,
        snr_out_db=snr_out,
        gain_db=snr_out - snr_in,
        noise_reduction_db=noise_reduction,
        bpm_in=core.estimate_bpm(voltage, fs),
        bpm_out=core.estimate_bpm(filtered, fs),
        true_bpm=true_bpm,
        n_beats_in=int(peaks_in.size),
        n_beats_out=int(peaks_out.size),
        correlation=correlation,
        r_peak_ratio=r_peak_ratio,
        std_ratio=std_ratio,
        rmse=error,
    )


def variant_table(
    voltage: np.ndarray,
    reference: np.ndarray | None,
    fs: float,
    low: float,
    high: float,
    q: float,
) -> pd.DataFrame:
    """Score the four standard variants so the app can show the full table.

    Args:
        voltage: Raw samples.
        reference: Ground truth, or ``None``.
        fs: Sampling rate in Hz.
        low: Band-pass lower cutoff in Hz.
        high: Band-pass upper cutoff in Hz.
        q: Notch quality factor.

    Returns:
        DataFrame with one row per variant.
    """
    if reference is None:
        return pd.DataFrame(columns=["variant", "snr_in_db", "snr_out_db", "gain_db", "bpm_after", "correlation"])
    rows = [
        core.evaluate(reference, voltage, fs, variant, low=low, high=high, q=q) for variant in core.VARIANTS
    ]
    return pd.DataFrame(rows)


DISPLAY_NAMES: dict[str, str] = {
    "variant": "Filter chain",
    "snr_in_db": "SNR in (dB)",
    "snr_out_db": "SNR out (dB)",
    "gain_db": "Gain (dB)",
    "bpm_before": "BPM before",
    "bpm_after": "BPM after",
    "correlation": "Correlation",
    "r_peak_ratio": "R-peak ratio",
    "std_ratio": "Std ratio",
    "rmse": "RMSE",
}
"""Readable headers for the metrics table."""


def display_table(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy of ``frame`` restricted to ``columns`` with readable headers.

    Args:
        frame: Metrics table (or any DataFrame with the known column names).
        columns: Column names to keep, in display order.

    Returns:
        A new DataFrame with the same values and friendlier headers.
    """
    return pd.DataFrame({DISPLAY_NAMES.get(name, name): np.asarray(frame[name]) for name in columns})
