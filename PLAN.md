# Build plan — ECG Signal Denoising (model + Streamlit app)

Repo: `/home/romeo/Projects/ECG-Signal_Analysis`
Python: use `.venv/bin/python` (streamlit 1.66.1, scipy 1.16.2, plotly 7.1.0, numpy 2.4.6)
**Never run `git commit` or `git push`.** Leave the working tree dirty for Thay to review.

## Part 1 — the data bug (blocking, fix this first)

`data/create_data.py` builds the ECG-like signal, adds the noise, then saves the **wrong array**:

```python
noisy_signal = clean_signal + noise_50Hz   # the real data
data_df = pd.DataFrame({'time': t, 'voltage': noise_50Hz})   # this is what got saved
```

Proven: filtering a pure 50 Hz sine reproduces the committed `data/process/clean_ecg_noisy.csv`
exactly (min/max ±0.07723, std 0.020036). So the committed "raw" is a sine wave and the
"cleaned" plot is Gibbs ringing — there is no P-QRS-T anywhere in the repo.

Fix `data/create_data.py` to:
1. Generate a **realistic synthetic ECG**, not two sines: sum of Gaussians for P, Q, R, S, T
   placed on a 60–75 bpm rhythm, ~10 s, 1000 Hz.
2. Add: 50 Hz powerline (amplitude 0.35), baseline wander (0.3 Hz, 0.15), white noise (σ 0.03).
3. Save **three** files:
   - `data/raw/ecg_noisy_raw.csv` → `time, voltage` (noisy)
   - `data/process/ecg_reference_clean.csv` → `time, voltage` (ground truth, for scoring)
   - `data/process/clean_ecg_noisy.csv` → `time, voltage_cleaned` (kept for backwards compat, produced by the filter)
4. Be deterministic (`np.random.default_rng(seed)`), so the repo reproduces.

Regenerate the committed CSVs and re-run the notebook so `images/graph_raw_data.png` and
`images/graph_clean_signal.png` show a real heartbeat, not a sine.

## Part 2 — the "model" (DSP pipeline + a real detector)

This project is signal processing, not ML — so the "model" is the filter chain plus a beat detector.

`src/fft_denoising.py` — rewrite as a reusable pipeline:

1. `spectrum(x, fs)` → freq, magnitude (rfft).
2. `notch(x, fs, f0, q)` → zero/attenuate the bins within ±1 Hz of `f0` (50 or 60), return signal.
3. `bandpass(x, fs, low=0.5, high=40)` → scipy Butterworth + `filtfilt` (zero-phase, no shift).
4. `denoise(x, fs, notch_hz, use_bandpass)` → composes the above.
5. `snr(reference, noisy)` → dB, using the clean reference as truth: `10*log10(var(ref)/var(noise))`.
6. `estimate_bpm(x, fs)` → simple Pan–Tompkins-lite: bandpass 5–15 Hz, square, moving-average
   integrate, adaptive threshold, refractory 200 ms → beat count → BPM.
7. `qrs_indices(x, fs)` → the peak locations, for the overlay.
8. A `run_all()` that writes a metrics table (SNR before/after for notch-only, bandpass-only, both)
   to `data/process/metrics.csv`.

Report real numbers. If the notch alone only gains a few dB, say so — bandpass + notch together
is what actually works, and that is a legitimate finding.

## Part 3 — the Streamlit app

```
app/
  Home.py                 # landing: what this is, the before/after SNR numbers, limitations
  lib/
    __init__.py
    io.py                 # load bundled CSV or user upload, schema check
    dsp.py                # thin wrappers over src/
    charts.py             # plotly builders
  pages/
    1_Signal_Viewer.py    # time domain: raw vs filtered, QRS markers
    2_Spectrum.py         # FFT magnitude, notch/bandpass controls, live re-filter
    3_Metrics.py          # SNR table, BPM before/after, what the filter costs (morphology check)
.streamlit/config.toml    # dark theme, clinical green/red accents
```

Requirements:
- `st.navigation` + `st.Page` (streamlit 1.66 API).
- Sidebar controls: sampling rate (default 1000), notch 50/60/off, bandpass on/off + low/high cutoffs,
  all wired to re-run the filter live on the selected record.
- File upload accepts a CSV with `time,voltage` (and tolerate `time,voltage_cleaned`), validates the
  schema and fails with a readable message, not a traceback.
- Show SNR and BPM as `st.metric` cards, before vs after.
- **Be explicit that the bundled signal is synthetic.** No clinical claims.
- `st.cache_data` on loaders; app starts under 2 s.
- No hardcoded absolute paths; works with `streamlit run app/Home.py` from the repo root.

## Acceptance checks (must all pass before saying it is done)

1. `.venv/bin/python -m compileall app src` → clean.
2. `pyright app src` → no errors.
3. `streamlit run app/Home.py --server.headless true --server.port 8602` starts and
   `curl -s localhost:8602/_stcore/health` returns `ok`.
4. Regenerated `data/process/clean_ecg_noisy.csv` is **not** a 0.077-amplitude sine: its std must be
   within ~15% of the reference signal's std (i.e. the heartbeat survived).
5. `snr()` on the regenerated pair reports a positive dB gain, and the same number appears in the app.
6. BPM estimated from the filtered signal is within ±3 of the BPM used to generate it.
7. No `print()` debug spam in the app.

## README

Rewrite with: what the app does, how to run it, the real SNR before/after table, the BPM result,
the honesty note that the bundled ECG is synthetic, and the `docs/` + `src/validation/` folders
that never existed — remove those claims or create the folders.
