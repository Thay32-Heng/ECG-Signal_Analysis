"""ECG Signal Analysis — landing page and navigation entrypoint.

Run from the repository root:

    .venv/bin/streamlit run app/Home.py
"""

from __future__ import annotations

import streamlit as st

from lib import charts, dsp, io as data_io, ui

st.set_page_config(
    page_title="ECG Signal Denoising",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home() -> None:
    """Render the landing page: headline numbers, method, honest findings."""
    st.title("ECG signal denoising")
    st.markdown(
        "A 50 Hz notch filter plus a zero-phase Butterworth band-pass clean a single-lead ECG, "
        "scored against the known clean reference. A Pan-Tompkins-lite detector then recovers the heart rate."
    )
    st.warning(ui.DISCLAIMER, icon=":material/science:")

    try:
        record = data_io.load_bundled()
    except (data_io.DataValidationError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()

    settings = dsp.FilterSettings(fs=record.fs, notch_hz=50.0, use_bandpass=True, low=0.5, high=40.0, q=35.0)
    filtered = dsp.apply_filter(
        record.voltage,
        settings.fs,
        settings.notch_hz,
        settings.use_bandpass,
        settings.low,
        settings.high,
        settings.q,
    )
    metrics = dsp.measure(record.voltage, filtered, record.reference, record.fs, record.true_bpm)
    table = dsp.variant_table(record.voltage, record.reference, record.fs, settings.low, settings.high, settings.q)

    st.subheader("Bundled record at a glance")
    ui.metric_cards(metrics, settings)

    window = min(4.0, record.duration_s)
    visible = record.time <= window
    peaks = dsp.detect_qrs(filtered, record.fs)
    peak_times = record.time[peaks]
    peak_values = filtered[peaks]
    in_window = peak_times <= window

    st.plotly_chart(
        charts.comparison_figure(
            record.time[visible],
            record.voltage[visible],
            filtered[visible],
            reference=record.reference[visible] if record.reference is not None else None,
            peak_times=peak_times[in_window],
            peak_values=peak_values[in_window],
            height=520,
        ),
        width="stretch",
    )
    st.caption(f"First {window:.0f} s of the record. Dotted grey line: the ground truth the filters never see.")

    st.subheader("What the numbers actually say")
    left, right = st.columns([3, 2])
    with left:
        if not table.empty:
            st.dataframe(
                dsp.display_table(
                    table,
                    ["variant", "snr_in_db", "snr_out_db", "gain_db", "bpm_after", "correlation", "r_peak_ratio"],
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No clean reference is available, so the SNR table cannot be computed for this record.")
    with right:
        if not table.empty:
            notch_gain = float(table.loc[table["variant"] == "notch_only", "gain_db"].iloc[0])
            band_gain = float(table.loc[table["variant"] == "bandpass_only", "gain_db"].iloc[0])
            both_gain = float(table.loc[table["variant"] == "notch+bandpass", "gain_db"].iloc[0])
            peak_ratio = float(table.loc[table["variant"] == "notch+bandpass", "r_peak_ratio"].iloc[0])
            st.markdown(
                f"""
- The notch on its own buys **{notch_gain:+.1f} dB**.
- The band-pass on its own buys **{band_gain:+.1f} dB**.
- Together they buy **{both_gain:+.1f} dB**. Neither filter is sufficient alone: a 40 Hz
  low-pass only partly attenuates 50 Hz mains hum, and a notch does nothing about wander.
- The cost shows up in the R-peak ratio (**{peak_ratio:.3f}**): the filter shaves a few percent
  off the peak height. That is the honest trade-off, not a free win.
"""
            )

    st.subheader("How the chain works")
    col_notch, col_band, col_detector = st.columns(3)
    col_notch.markdown(
        "**1 · FFT notch**\n\n"
        "Bins within ±1 Hz of 50 Hz are attenuated with a raised-cosine taper, not zeroed, "
        "so the reconstruction does not ring."
    )
    col_band.markdown(
        "**2 · Band-pass**\n\n"
        "Butterworth 0.5–40 Hz applied forwards and backwards (`sosfiltfilt`). Zero phase means "
        "the R peaks keep their timing."
    )
    col_detector.markdown(
        "**3 · QRS detector**\n\n"
        "5–15 Hz, derivative, square, 150 ms moving-window integration, adaptive threshold, "
        "200 ms refractory — Pan-Tompkins, simplified."
    )

    st.subheader("Limitations")
    st.markdown(
        """
- The bundled ECG is **synthetic** (sum of Gaussians on a 72 bpm rhythm). No patient data, no
  lead variability, no arrhythmias — the numbers here are a sanity check, not a benchmark.
- There is no machine learning in this project. The "model" is the filter chain plus the detector.
- The filters are scored against a reference that only exists because the noise was added
  synthetically. On a real recording this SNR cannot be computed without a clean lead or expert
  annotations.
- One lead, one rate, no motion artefact, and a baseline wander that is a single sine.
"""
    )

    st.subheader("Run it")
    st.code(
        "python data/create_data.py        # regenerate the synthetic record\n"
        "python src/fft_denoising.py       # score every filter chain -> data/process/metrics.csv\n"
        "streamlit run app/Home.py         # this app",
        language="bash",
    )
    st.caption("Use `.venv/bin/python` and `.venv/bin/streamlit` when the virtualenv is not activated.")
    ui.footer(record)


navigation = st.navigation(
    [
        st.Page(home, title="Overview", icon=":material/home:", url_path="overview", default=True),
        st.Page("pages/1_Signal_Viewer.py", title="Signal viewer", icon=":material/monitor_heart:"),
        st.Page("pages/2_Spectrum.py", title="Spectrum", icon=":material/insights:"),
        st.Page("pages/3_Metrics.py", title="Metrics", icon=":material/query_stats:"),
    ],
    position="sidebar",
)
navigation.run()
