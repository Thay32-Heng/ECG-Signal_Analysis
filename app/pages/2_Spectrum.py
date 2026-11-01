"""Frequency-domain view: what the notch and the band-pass actually remove."""

from __future__ import annotations

import numpy as np
import streamlit as st

from lib import charts, dsp, ui

record = ui.record_selector()
settings = ui.filter_controls(record)

filtered = dsp.apply_filter(
    record.voltage,
    settings.fs,
    settings.notch_hz,
    settings.use_bandpass,
    settings.low,
    settings.high,
    settings.q,
)
metrics = dsp.measure(record.voltage, filtered, record.reference, settings.fs, record.true_bpm)
freq, raw_magnitude, filtered_magnitude = dsp.spectrum_pair(record.voltage, filtered, settings.fs)

st.title("Spectrum")
st.caption("Single-sided amplitude spectrum, scaled so a pure sine of amplitude A peaks at A.")


def _format_amplitude(value: float) -> str:
    """Format a spectrum amplitude, switching to scientific notation when tiny.

    Args:
        value: Amplitude to format.

    Returns:
        A short display string.
    """
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def _amplitude_at(axis: np.ndarray, magnitude: np.ndarray, target: float) -> float:
    """Read the spectrum magnitude at the bin closest to ``target`` Hz.

    Args:
        axis: Frequency axis in Hz.
        magnitude: Amplitude spectrum.
        target: Frequency of interest in Hz.

    Returns:
        Amplitude at the nearest bin.
    """
    index = int(np.argmin(np.abs(axis - target)))
    return float(magnitude[index])


if settings.notch_hz is not None:
    before = _amplitude_at(freq, raw_magnitude, settings.notch_hz)
    after = _amplitude_at(freq, filtered_magnitude, settings.notch_hz)
    attenuation = 20.0 * np.log10(before / after) if after > 0 and before > 0 else float("nan")
    left, mid, right = st.columns(3)
    left.metric(f"{settings.notch_hz:.0f} Hz amplitude, raw", _format_amplitude(before))
    mid.metric(f"{settings.notch_hz:.0f} Hz amplitude, filtered", _format_amplitude(after))
    right.metric("Notch attenuation", f"{attenuation:.1f} dB", help="How far the mains spike dropped.")
else:
    st.info("The notch is switched off, so the mains spike is still there.")

st.plotly_chart(
    charts.spectrum_figure(
        freq,
        raw_magnitude,
        filtered_magnitude,
        notch_hz=settings.notch_hz,
        band=(settings.low, settings.high) if settings.use_bandpass else None,
        log_y=st.toggle("Logarithmic magnitude axis", value=False),
        max_hz=st.slider(
            "Show up to (Hz)",
            min_value=20.0,
            max_value=float(max(21.0, settings.fs / 2.0)),
            value=float(min(120.0, max(21.0, settings.fs / 2.0))),
            step=10.0,
        ),
    ),
    width="stretch",
)

st.subheader("Reading the plot")
left, right = st.columns(2)
with left:
    st.markdown(
        f"""
**The mains spike.** A clean 50 Hz sine shows up as a single narrow peak; the red shading marks the
notch band (±1 Hz). The taper means the bins just outside the band are barely touched, which is why
the filtered trace rejoins the raw one immediately.
"""
    )
with right:
    st.markdown(
        f"""
**The band-pass.** The green shading is the {settings.low:g}–{settings.high:g} Hz pass-band. Below
{settings.low:g} Hz sits the baseline wander (0.3 Hz in the bundled record); above {settings.high:g} Hz
sits muscle noise and the rest of the mains harmonics. Note that a {settings.high:g} Hz low-pass only
attenuates 50 Hz by a few dB — that is why the notch is still needed.
"""
    )

if settings.notch_hz is None and not settings.use_bandpass:
    st.caption(
        "No filter is active, so nothing has been removed — the raw and filtered "
        "traces are identical, and the removal figure is undefined."
    )
else:
    st.caption(
        f"The active chain removed {metrics.noise_reduction_db:.1f} dB of raw power, i.e. the removed "
        f"noise carried {100.0 * 10 ** (-metrics.noise_reduction_db / 10):.0f}% of it. "
        "That is why the raw record scores a negative SNR."
    )
ui.footer(record)
