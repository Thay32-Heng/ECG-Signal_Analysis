"""Plotly figure builders.

Plotly is imported only here so the data layer stays free of chart code. Every
figure uses the app's clinical palette: red for the contaminated trace, green
for the filtered one, amber for detected beats.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_RAW: str = "#EF4444"
COLOR_FILTERED: str = "#22C55E"
COLOR_REFERENCE: str = "#94A3B8"
COLOR_PEAK: str = "#F59E0B"
COLOR_GRID: str = "#1E293B"
COLOR_TEXT: str = "#E2E8F0"
COLOR_NOTCH: str = "rgba(239, 68, 68, 0.18)"
COLOR_BAND: str = "rgba(34, 197, 94, 0.12)"

MAX_PLOT_POINTS: int = 20_000
"""Upper bound on rendered points; longer records are decimated peak-preserving."""


def _decimate(time: np.ndarray, values: np.ndarray, max_points: int = MAX_PLOT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a trace to at most ``max_points`` samples without losing extremes.

    Each block keeps the sample with the largest absolute value, so R peaks
    survive decimation instead of being skipped by a plain stride.

    Args:
        time: Sample times.
        values: Signal values.
        max_points: Target number of points.

    Returns:
        Tuple ``(time, values)`` of equal, reduced length.
    """
    count = values.size
    if count <= max_points:
        return time, values
    step = int(np.ceil(count / max_points))
    usable = (count // step) * step
    blocks = values[:usable].reshape(-1, step)
    offsets = np.argmax(np.abs(blocks), axis=1)
    indices = offsets + np.arange(0, usable, step)
    return time[indices], values[indices]


def _style(fig: go.Figure, height: int, y_title: str, x_title: str) -> go.Figure:
    """Apply the shared dark layout.

    Args:
        fig: Figure to style.
        height: Height in pixels.
        y_title: Y-axis label.
        x_title: X-axis label.

    Returns:
        The same figure, styled.
    """
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text=x_title, gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=COLOR_GRID, zeroline=False)
    return fig


def comparison_figure(
    time: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    peak_times: np.ndarray | None = None,
    peak_values: np.ndarray | None = None,
    height: int = 520,
    max_points: int = MAX_PLOT_POINTS,
) -> go.Figure:
    """Build the stacked before/after figure used by the viewer and the home page.

    Args:
        time: Sample times in seconds.
        raw: Contaminated signal.
        filtered: Filtered signal.
        reference: Optional ground truth drawn behind the filtered trace.
        peak_times: Times of detected R peaks.
        peak_values: Amplitudes of detected R peaks.
        height: Figure height in pixels.
        max_points: Point budget per trace.

    Returns:
        A two-row Plotly figure sharing the time axis.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
    )

    raw_t, raw_y = _decimate(time, raw, max_points)
    fig.add_trace(
        go.Scattergl(x=raw_t, y=raw_y, mode="lines", name="Raw", line=dict(color=COLOR_RAW, width=1)),
        row=1,
        col=1,
    )

    if reference is not None:
        ref_t, ref_y = _decimate(time, reference, max_points)
        fig.add_trace(
            go.Scattergl(
                x=ref_t,
                y=ref_y,
                mode="lines",
                name="Reference (ground truth)",
                line=dict(color=COLOR_REFERENCE, width=1, dash="dot"),
            ),
            row=2,
            col=1,
        )

    filt_t, filt_y = _decimate(time, filtered, max_points)
    fig.add_trace(
        go.Scattergl(x=filt_t, y=filt_y, mode="lines", name="Filtered", line=dict(color=COLOR_FILTERED, width=1.4)),
        row=2,
        col=1,
    )

    if peak_times is not None and peak_values is not None and peak_times.size:
        fig.add_trace(
            go.Scatter(
                x=peak_times,
                y=peak_values,
                mode="markers",
                name=f"R peaks ({peak_times.size})",
                marker=dict(color=COLOR_PEAK, size=7, symbol="triangle-down"),
            ),
            row=2,
            col=1,
        )

    _style(fig, height, "Voltage (a.u.)", "Time (s)")
    # Two stacked panels: Plotly's subplot titles sit *inside* the plot area and
    # collide with the legend and the row-2 top spine, so each panel is labelled
    # through its own y-axis title instead.
    fig.update_yaxes(title_text="Raw (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="Filtered (a.u.)", row=2, col=1)
    # shared_xaxes puts the time label on the top row, which lands between the
    # two panels; it belongs under the bottom one.
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(margin=dict(l=70, r=10, t=56, b=10))
    return fig


def spectrum_figure(
    freq: np.ndarray,
    raw_magnitude: np.ndarray,
    filtered_magnitude: np.ndarray,
    *,
    notch_hz: float | None = None,
    band: tuple[float, float] | None = None,
    log_y: bool = False,
    max_hz: float = 120.0,
    height: int = 460,
) -> go.Figure:
    """Build the amplitude-spectrum figure with the filter bands shaded.

    Args:
        freq: Frequency axis in Hz.
        raw_magnitude: Amplitude spectrum of the raw signal.
        filtered_magnitude: Amplitude spectrum of the filtered signal.
        notch_hz: Mains frequency to shade, if any.
        band: ``(low, high)`` band-pass limits to shade, if any.
        log_y: Draw the magnitude axis on a log scale.
        max_hz: Upper limit of the frequency axis.
        height: Figure height in pixels.

    Returns:
        A Plotly figure.
    """
    limit = int(np.searchsorted(freq, max_hz))
    limit = max(limit, 2)

    fig = go.Figure()
    if band is not None:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor=COLOR_BAND, line_width=0, layer="below")
    if notch_hz is not None:
        fig.add_vrect(
            x0=max(0.0, notch_hz - 1.0),
            x1=notch_hz + 1.0,
            fillcolor=COLOR_NOTCH,
            line_width=0,
            layer="below",
            annotation_text=f"{notch_hz:.0f} Hz",
            annotation_position="top",
        )

    fig.add_trace(
        go.Scattergl(
            x=freq[:limit],
            y=raw_magnitude[:limit],
            mode="lines",
            name="Raw",
            line=dict(color=COLOR_RAW, width=1.2),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=freq[:limit],
            y=filtered_magnitude[:limit],
            mode="lines",
            name="Filtered",
            line=dict(color=COLOR_FILTERED, width=1.2),
        )
    )

    _style(fig, height, "Amplitude (a.u.)", "Frequency (Hz)")
    fig.update_yaxes(type="log" if log_y else "linear")
    return fig


def beat_figure(
    time: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    *,
    peak_times: np.ndarray | None = None,
    peak_values: np.ndarray | None = None,
    height: int = 300,
) -> go.Figure:
    """Build a single-panel overlay of the raw and filtered traces.

    Args:
        time: Sample times in seconds (already windowed).
        raw: Raw samples for the window.
        filtered: Filtered samples for the window.
        peak_times: Detected peak times inside the window.
        peak_values: Detected peak amplitudes inside the window.
        height: Figure height in pixels.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=time, y=raw, mode="lines", name="Raw", line=dict(color=COLOR_RAW, width=1))
    )
    fig.add_trace(
        go.Scattergl(x=time, y=filtered, mode="lines", name="Filtered", line=dict(color=COLOR_FILTERED, width=1.6))
    )
    if peak_times is not None and peak_values is not None and peak_times.size:
        fig.add_trace(
            go.Scatter(
                x=peak_times,
                y=peak_values,
                mode="markers",
                name="R peaks",
                marker=dict(color=COLOR_PEAK, size=9, symbol="triangle-down"),
            )
        )
    _style(fig, height, "Voltage (a.u.)", "Time (s)")
    return fig
