# Generating Raw Data
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

root = Path(__file__).resolve().parent

# Setup Parameter 
sampling_rate = 1000 # Hz
t = np.linspace(0, 1, sampling_rate) #Time from 0 to 1 second 

# Generate a Clean ECG-Like Signal (Simplified)
clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2* np.pi * 2 * t)

# Add 50Hz Power-Line Noise 
noise_50Hz = 0.5 * np.sin(2 * np.pi * 50 * t)
noisy_signal = clean_signal + noise_50Hz # This is our "Raw Data"

# Save to CSV 
data_df = pd.DataFrame({
    'time': t,
    'voltage': noise_50Hz
})
output_path = root / 'raw' / 'ecg_noisy_raw.csv'
data_df.to_csv(output_path, index = False) 

print("Raw noisy signal generate and saved to data/raw/:")