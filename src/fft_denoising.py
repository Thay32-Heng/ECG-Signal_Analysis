import pandas as pd 
import numpy as np 
from pathlib import Path

def denoise_ecg():
    # Setup Paths
    root = Path(__file__).resolve().parents[1]
    raw_path = root / 'data' / 'raw'/ 'ecg_noisy_raw.csv'
    processed_path = root / 'data' / 'process'/ 'clean_ecg_noisy.csv'

    # Load Raw Noisy Data
    df = pd.read_csv(raw_path)
    signal = df['voltage'].values
    sampling_rate = 1000 # Hz
    n = len(signal)

    # Compute FFT 
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, d=1/sampling_rate)

    # Filter 50Hz Noise
    idx_50hz = np.where(np.isclose(freq, 50, atol=1))[0]
    fft_spectrum[idx_50hz] = 0 # Remove noise frequency 

    # Reconstruct Clean Signal (IFFT)
    clean_signal = np.fft.irfft(fft_spectrum)

    # Save Cleaned Data
    processed_df = pd.DataFrame({
        'time': df['time'],
        'voltage_cleaned': clean_signal
    })
    processed_df.to_csv(processed_path, index= False)
    print("Data cleaned and saved to :", processed_path)

if __name__ == "__main__":
    denoise_ecg()