# This comes from the "jerks" algorithm as described in:
'''
Weaver, R. Glenn, et al.
"Jerks are useful: extracting pulse rate from wrist-placed accelerometry jerk during sleep in children."
Sleep 48.2 (2025): zsae099.
'''
# A matlab implementation was provided by the authors, this is a Python translation based on the appraoch
# described in the paper. We never compared this implementation directly to the matlab code, but the results
# seem to be similar.

import polars as pl
import numpy as np
import os
from scipy.signal import medfilt, hilbert, find_peaks, filtfilt

import matplotlib.pyplot as plt
from BCGAlgorithms.shared_functionalities import (
    filter_signal_butter, 
    STFT_from_window, 
    predict_from_stft, 
    get_hr_label,
    get_butter_sos,
    STFT_FLOAT_TYPE
)

from BCGAlgorithms.processor import BCGProcessor
from helpers.data_loader import load_test_data_nightbeatdb, load_test_data_oliviawalch, get_repo_path, load_transformed_data, get_base_paths

def root_mean_square_smoother(signal, m):
    # Optimized rms: signal^2 -> rolling sum (via convolution) -> mean -> sqrt
    sq = signal ** 2
    # Moving average
    kernel = np.ones(m) / m
    s = np.convolve(sq, kernel, mode='valid')
    # Pad to match original length (mimicking edge padding of original)
    pad_len = len(signal) - len(s)
    # Original implementation used custom rolling_window which usually dropped end
    # Here we pad end to match size
    # However, to strictly match original: 
    # rolling_window(signal, m) -> mean -> sqrt. 
    # The output size of original was (N - m + 1).
    return np.sqrt(s)

def moving_std(y, fs=100, window_size_seconds=0.1):
    m = int(fs * window_size_seconds)
    # Optimized moving std using convolution: E[X^2] - (E[X])^2
    kernel = np.ones(m) / m
    mean = np.convolve(y, kernel, 'valid')
    mean_sq = np.convolve(y**2, kernel, 'valid')
    
    var = mean_sq - mean**2
    # Fix potential negative zero errors
    var[var < 0] = 0
    std = np.sqrt(var)
    
    # Pad to match original output style if necessary? 
    # The original implementation returned the rolling std, likely valid length.
    return std, None 

def get_jerk(signal, jerk_diff=1, smoothing_window=18 / 50, fs=100, plot=False):
    jerk = np.diff(signal, jerk_diff) / fs
    jerk = np.pad(jerk, (0, jerk_diff), 'edge')

    # RMS smoothing
    win_len = int(smoothing_window * fs)
    jerk_s = root_mean_square_smoother(jerk, win_len)
    # Pad back to length
    jerk_s = np.pad(jerk_s, (0, len(signal) - len(jerk_s)), 'edge')
    jerk_s = jerk_s - np.mean(jerk_s)

    # Hilbert
    analytic = hilbert(jerk_s)
    jerk_h_angle = np.angle(analytic)
    
    # High pass (subtract moving average)
    win_hp = int(0.8 * fs)
    # Optimized using convolution
    smoothing = np.convolve(jerk_h_angle, np.ones(win_hp)/win_hp, 'same')
    
    # Original logic had custom padding logic, 'same' convolution approximates it closely
    # To exactly match "start_idx ... end_idx" logic from original would require manual indexing
    # But 'same' convolution is standard for this.
    jerk_h_angle = jerk_h_angle - smoothing

    return jerk_h_angle

def jerks_process_window(row, fs, nperseg=2**10, noverlap=2**10-20, padding_factor=10, sos=None):
    acc_a_mag = np.sqrt(np.array(row['acc_a_x'])**2 + np.array(row['acc_a_y'])**2 + np.array(row['acc_a_z'])**2)

    # ensure the length is a multiple of fs
    acc_a_mag = acc_a_mag[:(len(acc_a_mag) - len(acc_a_mag) % fs)]

    jerk = get_jerk(acc_a_mag, fs=fs)

    # Use pre-calculated SOS filter
    jerk_filtered = filter_signal_butter(jerk, sos=sos)

    magnitude, extent = STFT_from_window(jerk, fs, nperseg=nperseg, noverlap=noverlap, padding_factor=padding_factor)

    return {
        'magnitude': magnitude,
        'extent': extent,
        'jerk_filtered': jerk_filtered
    }

def jerks_load_and_process(subject, all_subjects = [], dataset='nightbeatdb', rows = 1000):
    fs = 100
    sos = get_butter_sos(0.5, 3.5, fs, order=4)
    
    output_schema = {
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'magnitude': pl.Array(pl.Array(STFT_FLOAT_TYPE, 946), 359),
        'extent': pl.List(pl.Float32),
        'jerk_filtered': pl.List(pl.Float64)
    }

    processor = BCGProcessor(
        algo_name='jerks',
        process_window_func=jerks_process_window,
        output_schema=output_schema,
        fs=fs
    )
    
    return processor.load_and_process(
        subject, all_subjects, dataset, rows,
        sos=sos
    )

def jerks_data_quality_metric(magnitude, rolling_size=120, fs = 100, t=[]):
    widths = []
    for i in range(magnitude.shape[1]):
        peaks, peak_properties = find_peaks(magnitude[:, i], width = 0.1)
        if len(peaks) == 0:
            widths.append(np.nan)
            continue
        peak_idx = np.argmax(peak_properties['prominences'])
        widths.append(peak_properties['widths'][peak_idx])

    widths = np.array(widths)

    rolling_size_adapted = len(np.where((t>=0) & (t<=rolling_size))[0])
    
    # Use optimized moving_std
    width_moving_std, _ = moving_std(widths, fs=1, window_size_seconds=rolling_size_adapted)

    len_diff = magnitude.shape[1] - len(width_moving_std)
    width_moving_std = np.pad(width_moving_std, (len_diff//2, len_diff//2 + len_diff % 2), 'edge')

    return width_moving_std

def jerks_predict_whole_window(row, fs, win_size, overlap=0.5, debug = False):
    assert win_size in [20, 30, 60], 'Window size not recognized'
    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []

    start_sec = 60
    magnitude = np.array(row['magnitude'][0])
    extent = row['extent'][0]
    t = np.linspace(extent[0], extent[1], magnitude.shape[1])
    f = np.linspace(extent[2], extent[3], magnitude.shape[0])
    
    sig_length = t[0] + t[-1]
    
    # Fix for missing acc_a_mag if using original logic that doesn't save it
    # Jerks saves 'jerk_filtered' but quality metric uses 'magnitude' in this specific file implementation
    data_quality = jerks_data_quality_metric(magnitude, int(2*(sig_length - 120 - win_size/2)), t=t)

    while start_sec < 120:
        end_sec = start_sec + win_size
        if debug:
            print(f"Predicting window {start_sec} to {end_sec}")

        t_index = np.where((t >= start_sec) & (t < end_sec))[0]
        label = get_hr_label(row, start_sec, end_sec)

        pred = predict_from_stft(magnitude[:, t_index], f, fs, debug, specific_algorithm='bioinsights')
        win_quality = np.nanmean(data_quality[t_index]) # Use nanmean for safety

        times_l.append(row['time'][0] + start_sec)
        preds_l.append(pred)
        labels_l.append(label)
        quality_l.append(win_quality)

        start_sec += int(win_size * (1 - overlap))

    return times_l, preds_l, labels_l, quality_l

def jerks_predict_df(df, fs=100, win_size=20, overlap=0.5):
    row_idxs_l = []
    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []

    for i in range(len(df)):
        row = df[i]
        row_idxs = row['row_idx'][0]
        times, preds, labels, quality = jerks_predict_whole_window(row, fs, win_size, overlap)

        row_idxs_l.extend([row_idxs] * len(times))
        times_l.extend(times)
        preds_l.extend(preds)
        labels_l.extend(labels)
        quality_l.extend(quality)

    df = pl.DataFrame({
        'row_idx': row_idxs_l,
        'time': times_l,
        'pred': preds_l,
        'label': labels_l,
        'quality': quality_l
    }, schema={
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'pred': pl.Float64,
        'label': pl.Float64,
        'quality': pl.Float64
    })
    return df

def jerks_load_and_predict(subject, all_subjects = [], dataset='nightbeatdb', win_size = 30):
    df = load_transformed_data(subject_id=all_subjects[subject], method='jerks', dataset=dataset)

    if dataset == 'nightbeatdb':
        df_orig = load_test_data_nightbeatdb(subject_id=all_subjects[subject], windows=1000)
    elif dataset == 'aw':
        df_orig = load_test_data_oliviawalch(subject_id=all_subjects[subject], windows=1000)

    df_orig = df_orig.with_row_index(name='row_idx').with_columns(pl.col('row_idx').cast(pl.Int64).alias('row_idx'))

    if 'start_100Hz' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start_100Hz').cast(pl.Float64).alias('time'))
    elif 'start' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start').cast(pl.Float64).alias('time'))

    df = df.join(df_orig, on=['row_idx', 'time'], how='inner', coalesce=True)
    if 'start_100Hz' in df.columns:
        df = df.filter(pl.col('hrv_is_valid').arr.get(1024 * 90) == 1)

    print(f"Predicting subject {all_subjects[subject]} in dataset {dataset}")

    df = jerks_predict_df(df, win_size=win_size)

    base_path = get_base_paths(results=True)[0 if dataset == 'nightbeatdb' else 1]
    out_name = f'jerks_{dataset}_{int(all_subjects[subject]):02d}_preds{win_size}.parquet'
    
    print(f"Saving in {os.path.join(base_path, out_name)}")
    df.write_parquet(os.path.join(base_path, out_name))

    return 'saved'

if __name__ == '__main__':
    jerks_load_and_process(0, all_subjects=range(32), dataset='wristbcg')