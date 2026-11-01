"""Record loading and schema validation.

Two sources are supported:

* the **bundled record** (``data/raw/ecg_noisy_raw.csv``) together with its
  clean reference (``data/process/ecg_reference_clean.csv``);
* a **user upload** of a CSV that must contain ``time`` and ``voltage``
  (``voltage_cleaned`` is accepted as a fallback so that files produced by the
  old pipeline still open).

Every failure path raises :class:`DataValidationError` with a message written
for a human, so the pages can show it with ``st.error`` instead of a traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_PATH: Path = REPO_ROOT / "data" / "raw" / "ecg_noisy_raw.csv"
REFERENCE_PATH: Path = REPO_ROOT / "data" / "process" / "ecg_reference_clean.csv"
METADATA_PATH: Path = REPO_ROOT / "data" / "process" / "ecg_metadata.json"

SIGNAL_COLUMNS: tuple[str, ...] = ("voltage", "voltage_cleaned")
"""Column names accepted as the signal, in order of preference."""

REFERENCE_COLUMNS: tuple[str, ...] = ("voltage_reference", "voltage_clean", "clean_voltage")
"""Column names accepted as ground truth when present in an upload."""

MIN_SAMPLES: int = 200
MAX_SAMPLES: int = 2_000_000
MAX_UPLOAD_MB: int = 25


class DataValidationError(ValueError):
    """Raised when a file cannot be interpreted as a single-lead ECG record."""


@dataclass(frozen=True)
class Record:
    """A loaded ECG record plus everything the pages need to describe it.

    Attributes:
        name: Human-readable source label.
        time: Sample times in seconds.
        voltage: Signal to filter (contaminated in the bundled record).
        fs: Sampling rate in Hz, inferred from the time column.
        reference: Ground truth when available, otherwise ``None``.
        is_synthetic: True for the bundled generated record.
        metadata: Generator metadata (heart rate, noise parameters, ...).
        ignored_columns: Columns that were present but not used.
    """

    name: str
    time: np.ndarray
    voltage: np.ndarray
    fs: float
    reference: np.ndarray | None
    is_synthetic: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    ignored_columns: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        """Record length in seconds."""
        return float(self.time[-1] - self.time[0])

    @property
    def true_bpm(self) -> float | None:
        """Generator heart rate when known, else ``None``."""
        value = self.metadata.get("realised_bpm") or self.metadata.get("target_bpm")
        return float(value) if isinstance(value, (int, float)) else None


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names so `` Voltage`` and ``voltage`` match."""
    return frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})


def _infer_fs(time: np.ndarray, source: str) -> float:
    """Infer the sampling rate from a time column and check uniform sampling.

    Args:
        time: Sample times in seconds.
        source: Label used in error messages.

    Returns:
        Sampling rate in Hz.

    Raises:
        DataValidationError: If the time axis is not strictly increasing and
            uniformly spaced.
    """
    if not np.all(np.isfinite(time)):
        raise DataValidationError(f"{source}: the 'time' column contains missing or infinite values.")
    steps = np.diff(time)
    if steps.size == 0 or np.any(steps <= 0):
        raise DataValidationError(f"{source}: the 'time' column must be strictly increasing.")
    mean_step = float(np.mean(steps))
    if float(np.std(steps)) / mean_step > 0.01:
        raise DataValidationError(
            f"{source}: the 'time' column is not uniformly sampled "
            f"(step varies by {100 * float(np.std(steps)) / mean_step:.1f}%). "
            "Resample the record to a constant rate before uploading."
        )
    return 1.0 / mean_step


def _validate_frame(frame: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, tuple[str, ...]]:
    """Validate an ECG table and split it into arrays.

    Args:
        frame: Raw parsed CSV.
        source: Label used in error messages.

    Returns:
        Tuple ``(time, voltage, reference, fs, ignored_columns)``.

    Raises:
        DataValidationError: If the table cannot be used as an ECG record.
    """
    if frame.empty:
        raise DataValidationError(f"{source}: the file has no rows.")

    table = _normalise(frame)
    if "time" not in table.columns:
        raise DataValidationError(
            f"{source}: no 'time' column found. Expected a CSV with columns 'time' and 'voltage'."
        )

    signal_column = next((name for name in SIGNAL_COLUMNS if name in table.columns), None)
    if signal_column is None:
        raise DataValidationError(
            f"{source}: no 'voltage' column found. Expected a CSV with columns 'time' and 'voltage'."
        )

    if len(table) < MIN_SAMPLES:
        raise DataValidationError(
            f"{source}: only {len(table)} samples found; at least {MIN_SAMPLES} are needed to filter and detect beats."
        )
    if len(table) > MAX_SAMPLES:
        raise DataValidationError(
            f"{source}: {len(table):,} samples exceed the {MAX_SAMPLES:,} sample limit for interactive use."
        )

    time = _as_float_array(table["time"])
    voltage = _as_float_array(table[signal_column])

    missing = int(np.count_nonzero(~np.isfinite(voltage)))
    if missing:
        raise DataValidationError(
            f"{source}: the '{signal_column}' column contains {missing} missing or non-numeric value(s). "
            "Fill or drop them before uploading."
        )

    reference_column = next((name for name in REFERENCE_COLUMNS if name in table.columns), None)
    reference = _as_float_array(table[reference_column]) if reference_column is not None else None
    if reference is not None and (reference.shape != voltage.shape or not np.all(np.isfinite(reference))):
        reference = None

    fs = _infer_fs(time, source)
    used = {"time", signal_column, reference_column}
    ignored = tuple(column for column in table.columns if column not in used)
    return time, voltage, reference, fs, ignored


def _as_float_array(column: object) -> np.ndarray:
    """Coerce a column to a float array, mapping unparsable values to ``nan``.

    Args:
        column: Column to convert (a pandas Series in practice).

    Returns:
        One-dimensional float array.
    """
    numeric = pd.to_numeric(column, errors="coerce")
    return np.asarray(numeric, dtype=float).reshape(-1)


def read_csv_bytes(raw: bytes, name: str) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame with a readable failure mode.

    Args:
        raw: Raw file content.
        name: File name, used in messages.

    Returns:
        Parsed table.

    Raises:
        DataValidationError: If the file is too large or cannot be parsed.
    """
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DataValidationError(f"{name}: file is larger than {MAX_UPLOAD_MB} MB.")
    try:
        return pd.read_csv(BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise DataValidationError(f"{name}: could not be parsed as CSV ({exc}).") from exc


@st.cache_data(show_spinner=False)
def load_bundled() -> Record:
    """Load the bundled synthetic record and its clean reference.

    Returns:
        The cached :class:`Record`.

    Raises:
        FileNotFoundError: If the generator has not been run yet.
    """
    if not RAW_PATH.exists() or not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Bundled data is missing. Run `.venv/bin/python data/create_data.py` from the repository root."
        )
    noisy = pd.read_csv(RAW_PATH)
    clean = pd.read_csv(REFERENCE_PATH)
    time, voltage, _, fs, _ = _validate_frame(noisy, "bundled record")
    reference = clean["voltage"].to_numpy(dtype=float)
    if reference.shape != voltage.shape:
        raise DataValidationError("Bundled reference length does not match the noisy record; regenerate the data.")
    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return Record(
        name="Bundled synthetic record",
        time=time,
        voltage=voltage,
        fs=fs,
        reference=reference,
        is_synthetic=True,
        metadata=metadata,
    )


@st.cache_data(show_spinner=False, max_entries=4)
def load_upload(raw: bytes, name: str) -> Record:
    """Validate and load a user-uploaded CSV.

    Args:
        raw: Raw file content.
        name: File name, used in messages and as the record label.

    Returns:
        The cached :class:`Record`.

    Raises:
        DataValidationError: If the file does not satisfy the schema.
    """
    frame = read_csv_bytes(raw, name)
    time, voltage, reference, fs, ignored = _validate_frame(frame, name)
    return Record(
        name=name,
        time=time,
        voltage=voltage,
        fs=fs,
        reference=reference,
        is_synthetic=False,
        metadata={},
        ignored_columns=ignored,
    )
