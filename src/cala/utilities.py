import numpy as np


def frequency_filter(arr: np.ndarray, cutoff_frequency: float, filter_pass: str):
    # Apply FFT-based filter
    _T = len(arr)
    cutoff_bin = int(cutoff_frequency * _T)

    # Perform real FFT
    frequency_composition = np.fft.rfft(arr)

    # Zero out the specified frequency bands
    if filter_pass == "low":
        frequency_composition[cutoff_bin:] = 0
    elif filter_pass == "high":
        frequency_composition[:cutoff_bin] = 0

    # Perform inverse real FFT to obtain the filtered signal
    filtered_arr = np.fft.irfft(frequency_composition, n=_T)
    return filtered_arr
