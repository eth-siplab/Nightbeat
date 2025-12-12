from scipy.signal import find_peaks, find_peaks_cwt, medfilt
import numpy as np
import polars as pl
from helpers.data_loader import get_base_paths

from BCGAlgorithms.shared_functionalities import rolling_window

import matplotlib.pyplot as plt

from BCGAlgorithms.shared_functionalities import filter_signal_fir, get_ibis

def get_narrow_peaks(x, pred, filter_width=0.1, kaiser_beta=1, filter_order=10, fs=100, robust_ibis=True, kernel_size_factor = 30, distance_factor=0.8, debug = False):

    low_cut = pred*(1-filter_width) / 60
    high_cut = pred*(1+filter_width) / 60

    # median filter
    kernel_size = int(kernel_size_factor/pred*60) + (int(kernel_size_factor/pred*60)+1)%2
    x_s = medfilt(x, kernel_size=kernel_size)

    x_s = np.mean(rolling_window(x_s, kernel_size), axis=1)
    length_diff = len(x) - len(x_s)
    x_s = np.pad(x_s, (length_diff//2, length_diff//2 + length_diff%2))

    y = filter_signal_fir(x_s, low_cut, high_cut, fs=fs, kaiser_beta=kaiser_beta, order=filter_order)

    kernel_size = int(50/pred*60) + (int(50/pred*60)+1)%2
    threshold = np.mean(rolling_window(y, kernel_size), axis=1)
    length_diff = len(y) - len(threshold)
    threshold = np.pad(threshold, (length_diff // 2, length_diff // 2 + length_diff % 2))

    peaks = find_peaks(y, height=threshold, distance=low_cut*fs*distance_factor)[0]

    ibis = get_ibis(peaks, fs=fs, robust=robust_ibis, lower_bound=1/high_cut*(1-filter_width), upper_bound = 1/low_cut*(1+filter_width))

    hr = 60/np.mean(ibis)

    if debug:
        plt.figure(figsize=(5, 3))
        plt.plot(x, label = 'x')
        plt.plot(x_s, label = 'x_s')
        plt.plot(y, label = 'y')
        plt.plot(threshold, label = 'threshold')
        plt.plot(peaks, y[peaks], "x")
        plt.legend()
        plt.xlim([500, 2000])
        plt.savefig(get_base_paths(results=True)[0]+ 'peak_detection_short.pdf')
        plt.show()

    return hr, y, peaks