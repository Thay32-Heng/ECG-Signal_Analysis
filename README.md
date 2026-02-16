# ECG Signal Denoising Using Fast Fourier Transform (FFT)

## Project Overview

[cite_start]This project focuses on the digital processing of Electrocardiogram (ECG) signals to remove environmental noise, specifically 50Hz power-line interference[cite: 21, 86]. [cite_start]By converting signals from the **Time Domain** to the **Frequency Domain** [cite: 24, 85][cite_start], we can accurately filter out artifacts while preserving vital cardiac waves (P, QRS, and T)[cite: 29, 172].

## Project Structure

Following professional data engineering best practices:

- `data/raw/`: Original ECG datasets containing noise.
- `data/processed/`: Cleaned ECG signals after FFT filtering.
- `docs/`: Presentation slides and technical documentation.
- `src/validation/`: Modules to verify signal improvement.
- [cite_start]`notebooks/`: Interactive analysis of FFT and IFFT stages[cite: 25].

## Methodology

1. [cite_start]**Acquisition**: Collecting raw ECG data contaminated with noise[cite: 36, 39].
2. [cite_start]**Transformation**: Applying **Fast Fourier Transform ($FFT$)** to identify noise spikes[cite: 29, 44].
3. [cite_start]**Filtering**: Targeting the $50Hz$ frequency component for removal[cite: 86, 95].
4. [cite_start]**Reconstruction**: Using **Inverse FFT ($IFFT$)** to restore the cleaned time-domain signal[cite: 29, 57].

## Results: Before vs. After

[cite_start]The primary metric for success is the visual and mathematical clarity of the signal[cite: 109].

| **Signal Phase** | **Visual Output**                              | **Key Characteristic**                                                          |
| :--------------- | :--------------------------------------------- | :------------------------------------------------------------------------------ |
| **Raw ECG**      | ![Noisy Signal](images/graph_raw_data.png)     | [cite_start]High-frequency $50Hz$ noise distortion[cite: 111, 146].             |
| **Filtered ECG** | ![Clean Signal](images/graph_clean_signal.png) | [cite_start]Preserved cardiac morphology for medical diagnosis[cite: 123, 172]. |

## Impact & Applications

- [cite_start]**Clinical Diagnostics**: Assisting doctors in detecting arrhythmias with higher accuracy[cite: 178, 179].
- [cite_start]**Wearable Tech**: Filtering motion artifacts in smartwatches[cite: 180, 181].
- [cite_start]**AI Cardiology**: Providing clean data for machine learning heart attack predictions[cite: 182, 183].

---

[cite_start]\*Developed DSE Year 2 (RUPP).
