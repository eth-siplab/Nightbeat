import numpy as np
import scipy.signal as signal
from scipy.signal import ShortTimeFFT, find_peaks
from scipy.stats import kurtosis, norm, iqr
import matplotlib.pyplot as plt
import polars as pl
import torch

# Check if Float16 is available in this Polars version
if hasattr(pl, "Float16"):
    # Float16 saves 50% memory compared to Float32
    STFT_FLOAT_TYPE = pl.Float16 
else:
    STFT_FLOAT_TYPE = pl.Float32

print(f"Using {STFT_FLOAT_TYPE} for Spectrogram storage.")

# --- OPTIMIZED FILTER FUNCTIONS ---

def get_butter_sos(lowcut, highcut, fs, order=5):
    """Calculate filter coefficients once."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def filter_signal_butter(y, lowcut=None, highcut=None, fs=None, order=5, sos=None):
    """
    Apply Butterworth filter. 
    Efficiency boost: Pass 'sos' directly to skip recalculation.
    """
    if sos is None:
        if lowcut is None or highcut is None or fs is None:
            raise ValueError("Must provide either 'sos' or 'lowcut', 'highcut', and 'fs'")
        sos = get_butter_sos(lowcut, highcut, fs, order)
        
    y = signal.sosfilt(sos, y)
    return y
def lowpass_it_pls(my_sig,fs,cuttoff):
    nyquist = 0.5 * fs
    normal_cutoff = cuttoff / nyquist
    b, a = signal.butter(3, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b,a,my_sig)

def filter_signal_fir(y, lowcut, highcut, fs, order=100, kaiser_beta =2, return_frequncy_response=False):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b = signal.firwin(order, [low, high], pass_zero=False, window=('kaiser', kaiser_beta))
    w, h = signal.freqz(b, 1, worN=2 * fs, fs=fs)
    y = np.convolve(y, b, mode='same')
    if return_frequncy_response:
        return y, w, h
    else:
        return y

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def detect_peaks(y, threshold=None, min_dist=30, ma_window=20):
    if threshold is None:
        threshold = np.mean(y)
    if isinstance(threshold, float):
        threshold = np.ones(len(y)) * threshold
    if isinstance(threshold, str):
        if threshold == 'MA':
            threshold = np.mean(rolling_window(y, ma_window), -1)
            len_diff = len(y) - len(threshold)
            threshold = np.pad(threshold, (len_diff//2, len_diff//2 + len_diff % 2), 'edge')
    peaks, _ = signal.find_peaks(y, height=threshold, distance=min_dist)
    return peaks

def get_ibis(peaks, fs = 100, robust = True, return_ibi_indexes = False, lower_bound = None, upper_bound = None):
    ibis = np.diff(peaks) / fs
    ibi_index = np.ones(len(ibis)).astype(bool)
    if lower_bound is not None:
        ibi_index = ibi_index & (ibis > lower_bound)
    if upper_bound is not None:
        ibi_index = ibi_index & (ibis < upper_bound)
    if robust:
        ibi_index_2 = get_robust_ibis(ibis[ibi_index])
        ibis = ibis[ibi_index][ibi_index_2]
        ibi_index_final = np.zeros(len(ibi_index)).astype(bool)
        ibi_index_final[np.where(ibi_index)[0][ibi_index_2]] = True
    else:
        ibi_index_final = ibi_index
        ibis = ibis[ibi_index_final]
    if return_ibi_indexes:
        return ibis, ibi_index_final
    else:
        return ibis

def get_robust_ibis(ibis, outlier_factor=3):
    ibi_median = np.median(ibis)
    ibi_iqr = iqr(ibis)
    if ibi_iqr == 0: return np.ones(len(ibis), dtype=bool)
    ibi_rv = norm(ibi_median, ibi_iqr / 1.34896)
    ibi_index = (0.5 - np.abs(ibi_rv.cdf(ibis) - 0.5)) > norm().cdf(-outlier_factor)
    return ibi_index

def get_hr_label(row, start_sec, end_sec, return_beat_hr=False, robust = False, debug = False):
    if 'beats_m' in row:
        beats = row['beats_m'][0].to_numpy()
        fs = 1024
    elif 'ppg_peaks' in row:
        beats = row['ppg_peaks'][0].to_numpy()
        fs = 500
    elif 'hr_vals' in row:
        hr_vals = row['hr_vals'][0].to_numpy()
        hr_times = row['hr_times'][0].to_numpy()
        
    if 'beats_m' in row or 'ppg_peaks' in row:
        beats = beats[(beats > start_sec * fs) & (beats < end_sec * fs)]
        beat_hr = 60 * len(beats) / (end_sec - start_sec)
        if len(beats) > 1:
            ibis = get_ibis(beats, fs, robust=robust, lower_bound=0.3, upper_bound=2)
            ibi_hr = 60 / np.mean(ibis) if len(ibis) > 0 else beat_hr
        else:
            ibi_hr = beat_hr
    elif 'hr_vals' in row:
        ibi_hr = np.mean(hr_vals[(hr_times > start_sec) & (hr_times < end_sec)])
        beat_hr = ibi_hr 

    if return_beat_hr:
        return ibi_hr, beat_hr
    else:
        return ibi_hr

def autocorrelation_torch(signal, device = 'cpu'):
    if not torch.is_tensor(signal):
        signal = torch.tensor(signal)
    if signal.get_device() < 0 and device != 'cpu':
        signal = signal.to(device)
    signal_mean = torch.mean(signal)
    signal_centered = signal - signal_mean
    autocorr = -torch.nn.functional.conv1d(signal_centered.view(1, 1, -1), signal_centered.view(1, 1, -1).flip(dims=[2]), padding=signal_centered.size(0) - 1)
    autocorr = autocorr.view(-1)[autocorr.size(2) // 2:]
    return autocorr / autocorr[0]

def STFT_from_window(x, fs, nperseg, noverlap, padding_factor, fft_mode='centered', scale_to='magnitude', phase_shift=None):
    win = ('kaiser', nperseg // 512)
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, mfft=nperseg * padding_factor, fft_mode=fft_mode,
                                   scale_to=scale_to, phase_shift=phase_shift)
    Zxx = SFT.stft(x)
    extent = SFT.extent(len(x), center_bins=True)
    f = np.linspace(extent[2], extent[3], Zxx.shape[0])
    t = np.linspace(extent[0], extent[1], Zxx.shape[1])
    
    # Filter frequencies of interest immediately
    mask = (f > 0) & (f < 3.5)
    magnitude = np.abs(Zxx)[mask, :]
    f = f[mask]
    extent = (t[0], t[-1], f[0], f[-1])
    return magnitude, extent

def normalize_signal(signal, signal_mean=None, signal_std=None):
    if signal_mean is None: signal_mean = np.mean(signal)
    if signal_std is None: signal_std = np.std(signal)
    if signal_std != 0: return (signal - signal_mean) / signal_std
    return signal - signal_mean

def predict_from_stft(magnitude, f, fs=100, debug=False, specific_algorithm=None):
    if specific_algorithm in ['bioinsights', 'jerks']:
        freq_dist = np.sum(magnitude, axis=1)
        # Assuming f matches magnitude rows
        peak_freq = f[(f > 0.5) & (f < 3.5)][np.argmax(freq_dist[(f > 0.5) & (f < 3.5)])]
    elif specific_algorithm == 'pwr':
        peak_freq = f[np.argmax(np.sum(magnitude, axis=1))]
    else:
        peak_freq = f[np.argmax(np.sum(magnitude, axis=1))]
    return 60 * peak_freq

def get_kurtosis_response(signal: np.array, fs: int, window_size: int = 10, overlap: int = 5):
    kurtosis_response = []
    for i in range(0, len(signal) - window_size * fs, overlap * fs):
        signal_minus_window = np.concatenate([signal[:i], signal[i + window_size * fs:]])
        kurtosis_response.append(kurtosis(signal_minus_window))
    return np.array(kurtosis_response)