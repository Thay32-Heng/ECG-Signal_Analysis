"""Metrics: SNR per filter chain, heart-rate accuracy and what filtering costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib import dsp, ui

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

st.title("Metrics")
st.caption(f"Active chain: {settings.label}")
ui.metric_cards(metrics, settings)

st.subheader("Heart rate")
first, second, third, fourth = st.columns(4)
first.metric("BPM from raw", f"{metrics.bpm_in:.1f}" if np.isfinite(metrics.bpm_in) else "n/a")
second.metric("BPM from filtered", f"{metrics.bpm_out:.1f}" if np.isfinite(metrics.bpm_out) else "n/a")
if metrics.true_bpm is not None:
    third.metric("BPM used to generate", f"{metrics.true_bpm:.1f}")
    error = metrics.bpm_error
    fourth.metric(
        "Absolute error",
        f"{error:.2f} bpm" if error is not None else "n/a",
        delta="within ±3 bpm" if error is not None and error <= 3.0 else "outside ±3 bpm",
        delta_color="normal" if error is not None and error <= 3.0 else "inverse",
    )
else:
    third.metric("Beats detected", f"{metrics.n_beats_out}")
    fourth.metric("Beats in raw", f"{metrics.n_beats_in}")

st.caption(
    "The detector band-passes to 5–15 Hz on its own, which is why the heart rate survives even in "
    "the raw record — the noise sits outside the QRS band. That is a property of the detector, not "
    "of the cleaning."
)

st.subheader("Signal-to-noise ratio by filter chain")
if record.reference is not None:
    table = dsp.variant_table(record.voltage, record.reference, settings.fs, settings.low, settings.high, settings.q)
    st.dataframe(
        dsp.display_table(
            table,
            [
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
            ],
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download metrics CSV",
        data=table.to_csv(index=False),
        file_name="ecg_metrics.csv",
        mime="text/csv",
        key="metrics_download",
    )
    st.caption(
        "SNR is computed against the clean reference: 10·log₁₀(var(reference) / var(reference − estimate)). "
        "The band-pass alone cannot reach the mains spike; the notch alone cannot reach the wander. "
        "The last row is the chain the app uses by default."
    )
else:
    if settings.notch_hz is None and not settings.use_bandpass:
        st.info(
            "This record has no clean reference and no filter is active, so there is "
            "nothing to compare. Switch on the notch or the band-pass first."
        )
    else:
        st.info(
            "This record has no clean reference, so SNR cannot be computed — SNR needs ground truth. "
            f"The filter removed {metrics.noise_reduction_db:.1f} dB of the raw signal power, which is a "
            "relative measure and is not the same thing as an SNR gain."
        )

st.subheader("What the filter costs")
if record.reference is not None:
    cost = pd.DataFrame(
        [
            {
                "Check": "Correlation with ground truth",
                "Value": metrics.correlation,
                "Ideal": "1.0",
                "Meaning": "1.0 = the morphology is untouched.",
            },
            {
                "Check": "R-peak height ratio",
                "Value": metrics.r_peak_ratio,
                "Ideal": "1.0",
                "Meaning": "Below 1.0 = the filter shaved the peak.",
            },
            {
                "Check": "Standard deviation ratio",
                "Value": metrics.std_ratio,
                "Ideal": "1.0",
                "Meaning": "Below 1.0 = overall amplitude was reduced.",
            },
            {
                "Check": "RMSE vs reference",
                "Value": metrics.rmse,
                "Ideal": "0.0",
                "Meaning": "Residual error in voltage units.",
            },
            {
                "Check": "Noise power vs raw power",
                "Value": metrics.noise_reduction_db,
                "Ideal": "—",
                "Meaning": (
                    "10·log₁₀(raw power / removed power); undefined when no filter "
                    "is active and nothing is removed."
                    if np.isnan(metrics.noise_reduction_db)
                    else (
                        "10·log₁₀(raw power / removed power); "
                        f"the removed noise carried {100.0 * 10 ** (-metrics.noise_reduction_db / 10):.0f}% "
                        "of the raw power."
                    )
                ),
            },
        ]
    )
    st.dataframe(cost, width="stretch", hide_index=True)
    st.markdown(
        f"""
The R peak is **{100 * (1 - metrics.r_peak_ratio):.1f}%** shorter after filtering and the trace
still correlates **{metrics.correlation:.3f}** with the ground truth. A 40 Hz low-pass rounds the
sharpest part of the QRS, and a 0.5 Hz high-pass trims the slow T wave — that is the price of
removing the noise. Loosening the cutoffs recovers amplitude but lets more artefact back in; try it
in the sidebar and watch the SNR fall.
"""
    )
else:
    st.info("Upload a CSV with a `voltage_reference` column to see the morphology cost table.")

ui.download_button(record.time, filtered, settings, key="metrics_page_download")
ui.footer(record)
