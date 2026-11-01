"""Time-domain view: raw vs filtered ECG with detected R peaks."""

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

st.title("Signal viewer")
st.caption(f"{settings.label} · {settings.fs:.0f} Hz")
ui.metric_cards(metrics, settings)

if record.duration_s > 1.5:
    first, second, third = st.columns([2, 2, 1.4])
    start = first.slider(
        "Window start (s)",
        min_value=0.0,
        max_value=float(np.floor(record.duration_s - 0.5)),
        value=0.0,
        step=0.5,
    )
    length = second.slider(
        "Window length (s)",
        min_value=0.5,
        max_value=float(min(10.0, record.duration_s)),
        value=float(min(4.0, record.duration_s)),
        step=0.5,
    )
    show_reference = third.toggle("Show ground truth", value=record.reference is not None)
else:
    start, length, show_reference = 0.0, record.duration_s, record.reference is not None

stop = min(start + length, record.duration_s)
window = (record.time >= start) & (record.time <= stop)
if not np.any(window):
    st.warning("The selected window is empty. Move the start slider back.")
    st.stop()

peaks = dsp.detect_qrs(filtered, settings.fs)
peak_times = record.time[peaks]
peak_values = filtered[peaks]
in_window = (peak_times >= start) & (peak_times <= stop)

overlay = st.toggle("Overlay both traces in one panel", value=False)

if overlay:
    figure = charts.beat_figure(
        record.time[window],
        record.voltage[window],
        filtered[window],
        peak_times=peak_times[in_window],
        peak_values=peak_values[in_window],
    )
else:
    figure = charts.comparison_figure(
        record.time[window],
        record.voltage[window],
        filtered[window],
        reference=record.reference[window] if (record.reference is not None and show_reference) else None,
        peak_times=peak_times[in_window],
        peak_values=peak_values[in_window],
    )

st.plotly_chart(figure, width="stretch")

intervals = np.diff(peak_times[in_window])
rr_text = f" · median RR {np.median(intervals) * 1000:.0f} ms" if intervals.size else ""
st.caption(
    f"{int(in_window.sum())} beats in the window "
    f"({int(peaks.size)} in the whole record){rr_text}. "
    "Amber triangles mark the R peaks the detector found on the filtered trace."
)

with st.expander("Why the R peaks do not move"):
    st.markdown(
        "`sosfiltfilt` runs the Butterworth filter forwards and then backwards, which cancels the "
        "phase response exactly. A one-way `lfilter` would delay the signal by several milliseconds "
        "and shift every detected beat; the zero-phase version keeps the peak timing intact."
    )

ui.download_button(record.time, filtered, settings, key="viewer_download")
ui.footer(record)
